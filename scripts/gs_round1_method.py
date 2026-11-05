#!/usr/bin/env python3
"""Run an official training function, capture efficiency, and render round-one views.

The training equations and optimizers are unmodified. Hooks retain the live scene
for auxiliary checkpoint serialization and time DoF-Gaussian's depth-prior call.
Each invocation runs in that method's independent environment.
"""
import argparse
import fcntl
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
REPOS = {'dofgs': 'DOFGS_implementation', 'dofgaussian': 'DoF-Gaussian',
         'mipsplatting': 'mip-splatting', 'cocogaussian': 'CoCoGaussian'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def guard():
    if shutil.disk_usage('/').free < 1024**3:
        raise RuntimeError('STOP: system disk below 1 GiB')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'


def data_view(source, destination, low_resolution=False):
    """Keep reader-generated PLYs and other writes outside the input dataset."""
    destination.mkdir(parents=True, exist_ok=True)
    image_source = source/('images_4' if low_resolution else 'images')
    for name, target in [('images', image_source), ('images_4', source/'images_4'),
                         ('poses_bounds.npy', source/'poses_bounds.npy'),
                         ('manifest.json', source/'manifest.json'), ('dense', source/'dense')]:
        link = destination/name
        if target.exists() and not link.exists():
            link.symlink_to(target.resolve(), target_is_directory=target.is_dir())
    if not (destination/'sparse').exists():
        shutil.copytree(source/'sparse', destination/'sparse')
        for path in (destination/'sparse').rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o644)
    for path in source.glob('hold=*'):
        (destination/path.name).touch(exist_ok=True)
    write(destination/'source.json', {'source': str(source), 'low_resolution_images': low_resolution,
                                    'geometry': 'byte-identical sparse copy; images are read-only links'})
    return destination


def parameters(job, source, model):
    from argparse import ArgumentParser, Namespace
    if job['method'] == 'dofgs':
        from omegaconf import OmegaConf
        from src.model_dofgs.arguments import ModelParams, PipelineParams, OptimizationParams
        filename = 'config_real_'+(job['scene'] if job['dataset'] in {'deblurnerf', 'author-retrained'} else 'cake')+'.yaml'
        saved = Path(job.get('model', ''))/'config_runtime.yaml'
        config = OmegaConf.load(saved if job.get('kind') == 'author' and saved.is_file() else WORKSPACE/REPOS['dofgs']/'config_files'/filename)
        config.source_path = str(source); config.model_path = str(model)
        config.hold = job['hold']; config.eval = True
        config.resolution = 1 if job['dataset'] == 'dofnerf-mix' else 4
        parser = ArgumentParser(); lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
        args = parser.parse_args([])
        for key, value in OmegaConf.to_container(config, resolve=True).items():
            setattr(args, key, value)
        args.debug_from = -1
        args.test_iterations = list(config.test_iterations)
        args.save_iterations = list(config.save_iterations)
        args.checkpoint_iterations = []
        args.start_checkpoint = None
        args.quiet = False
        if job.get('diagnostic_iterations'):
            args.iterations = job['diagnostic_iterations']; args.test_iterations = []; args.save_iterations = [args.iterations]; args.checkpoint_iterations = []
        return args, lp, op, pp
    from arguments import ModelParams, PipelineParams, OptimizationParams
    parser = ArgumentParser(); lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    args = parser.parse_args([])
    args.source_path = str(source); args.model_path = str(model)
    args.resolution = 1 if job['dataset'] == 'dofnerf-mix' else 4
    args.images = 'images'; args.eval = True; args.llffhold = job['hold']
    args.debug_from = -1; args.start_checkpoint = None; args.quiet = False
    args.test_iterations = list(range(3000, 30001, 3000)) if job['method'] == 'cocogaussian' else [7000, 30000]
    args.save_iterations = args.test_iterations.copy()
    args.checkpoint_iterations = [10000, 20000, 30000] if job['method'] == 'cocogaussian' else []
    if job.get('diagnostic_iterations'):
        args.iterations = job['diagnostic_iterations']; args.test_iterations = []; args.save_iterations = [args.iterations]; args.checkpoint_iterations = []
    return args, lp, op, pp


class Sampler:
    def __init__(self):
        self.stage = 'gaussian'; self.samples = []; self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop.is_set():
            try:
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
                memory = sum(float(row.split(',')[1]) for row in rows.splitlines() if int(row.split(',')[0]) == os.getpid())
                utilization = float(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip())
                self.samples.append({'time': time.time(), 'stage': self.stage, 'memory_mib': memory, 'gpu_utilization': utilization})
            except (ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.5)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set(); self.thread.join()
        return {stage: {'nvidia_peak_mib': max([r['memory_mib'] for r in self.samples if r['stage'] == stage] or [0]),
                        'mean_gpu_utilization': sum(r['gpu_utilization'] for r in self.samples if r['stage'] == stage)/max(1, sum(r['stage'] == stage for r in self.samples))}
                for stage in {'gaussian', 'depth_prior'}}


def train(job):
    import torch
    source = data_view(Path(job['source']), Path(job['output'])/'input_scene', job['dataset'] == 'dofnerf-mix')
    output = Path(job['output']); output.mkdir(parents=True, exist_ok=True)
    args, lp, op, pp = parameters(job, source, output)
    module = importlib.import_module('src.model_dofgs.training' if job['method'] == 'dofgs' else 'train')
    module.args = args  # DoF-Gaussian's official function reads its CLI namespace.
    utils = importlib.import_module('src.utils.general_utils' if job['method'] == 'dofgs' else 'utils.general_utils')
    utils.safe_state(False)
    holder = {}
    original_init = module.Scene.__init__
    def retain_scene(self, *a, **kw):
        original_init(self, *a, **kw); holder['scene'] = self
    module.Scene.__init__ = retain_scene
    sampler = Sampler(); prior_seconds = 0.; prior_peak = 0; pre_prior_peak = 0
    if job['method'] == 'dofgaussian':
        original_prior = module.depth_priors.train
        def timed_prior(*a, **kw):
            nonlocal prior_seconds, prior_peak, pre_prior_peak
            torch.cuda.synchronize(); pre_prior_peak = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats(); sampler.stage = 'depth_prior'; started = time.perf_counter()
            try:
                return original_prior(*a, **kw)
            finally:
                torch.cuda.synchronize(); prior_seconds += time.perf_counter()-started
                prior_peak = torch.cuda.max_memory_allocated(); torch.cuda.reset_peak_memory_stats(); sampler.stage = 'gaussian'
        module.depth_priors.train = timed_prior
    write(output/'arguments.json', vars(args))
    (output/'cfg_args').write_text(str(argparse.Namespace(**vars(lp.extract(args)))))
    info = {'job': job, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'efficiency_valid': bool(job.get('exclusive')) and not job.get('diagnostic_iterations'), 'timing_boundary': 'official training call, includes its data/model setup, official validation and saving; depth-prior call timed separately'}
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); sampler.start(); started = time.perf_counter()
    try:
        if job['method'] == 'dofgs':
            from src.model_dofgs.gaussian_model import GaussianModel
            module.training(GaussianModel, lp.extract(args), op.extract(args), pp.extract(args),
                args.test_iterations, args.save_iterations, args.checkpoint_iterations,
                args.add_point_iterations, None, -1, None, None)
        else:
            module.training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
                            args.save_iterations, args.checkpoint_iterations, None, -1)
        torch.cuda.synchronize()
        info.update(status='trained', total_training_call_seconds=time.perf_counter()-started,
                    depth_prior_seconds=prior_seconds,
                    gaussian_seconds=time.perf_counter()-started-prior_seconds,
                    torch_gaussian_peak_bytes=max(pre_prior_peak, torch.cuda.max_memory_allocated()),
                    torch_depth_prior_peak_bytes=prior_peak)
    except BaseException as exc:
        info.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), total_training_call_seconds=time.perf_counter()-started)
    finally:
        info['sampling'] = sampler.finish(); write(output/'training-efficiency.json', info)
        write(output/'gpu-samples.json', sampler.samples)
        module.Scene.__init__ = original_init
    if info['status'] != 'trained':
        raise RuntimeError(info['error'])
    scene = holder['scene']; gaussians = scene.gaussians
    assert torch.isfinite(gaussians.get_xyz).all(), 'Non-finite optimized positions'
    metadata = {'method': job['method'], 'source': str(source), 'original_source': job['source'],
                'dataset': job['dataset'], 'scene': job['scene'], 'iteration': args.iterations,
                'train_cameras': [{'name': c.image_name, 'uid': c.uid} for c in scene.getTrainCameras()],
                'test_cameras': [{'name': c.image_name, 'uid': c.uid} for c in scene.getTestCameras()],
                'gaussians': len(gaussians.get_xyz), 'gaussian_parameters': 59*len(gaussians.get_xyz),
                'official_config': vars(args), 'auxiliary_serialization': 'after the timed official training call'}
    if job['method'] == 'cocogaussian':
        metadata['focal_planes_in_optimizer'] = any(p is gaussians.CoCoBlurKernel.focal_planes for group in gaussians.optimizer.param_groups for p in group['params'])
        metadata['focal_plane_note'] = 'The public kernel initializes focal_planes lazily. Membership is recorded without changing the official optimizer.'
        torch.save({'state_dict': gaussians.CoCoBlurKernel.state_dict(), 'max_depth': gaussians.CoCoBlurKernel.max_depth,
                    'options': vars(op.extract(args))}, output/'coco-kernel.pt')
    if job['method'] == 'dofgaussian':
        torch.save({'focus': gaussians.get_disp_focus.detach(), 'K': gaussians.get_K_bokeh.detach()}, output/'lens-parameters.pt')
    metadata['model_bytes'] = sum(p.stat().st_size for p in (output/'point_cloud'/f'iteration_{args.iterations}').rglob('*') if p.is_file())
    optimized = {id(p): p for group in gaussians.optimizer.param_groups for p in group['params']}
    metadata['optimized_parameters'] = sum(p.numel() for p in optimized.values())
    auxiliary = [output/'coco-kernel.pt', output/'lens-parameters.pt', output/'defocus_params'/f'iteration_{args.iterations}'/'defocus_params.pth']
    metadata['auxiliary_bytes'] = sum(p.stat().st_size for p in auxiliary if p.is_file())
    metadata['model_bytes'] += metadata['auxiliary_bytes']
    write(output/'model-metadata.json', metadata)


def load_model(job):
    import torch
    model = Path(job.get('model', job['output']))
    metadata = model/'model-metadata.json'
    if metadata.exists():
        meta = json.loads(metadata.read_text()); args = argparse.Namespace(**meta['official_config'])
        source = Path(meta['source'])
    else:
        source = data_view(Path(job['source']), Path(job['output'])/'input_scene')
        args, _, _, _ = parameters(job, source, model)
        meta = {'original_source': job['source'], 'dataset': job['dataset']}
    args.model_path = str(model); args.source_path = str(source); args.eval = True
    if job['method'] == 'dofgs':
        from src.model_dofgs.gaussian_model import GaussianModel
        from src.scene import Scene
        from src.model_dofgs.renderer import render, ToneMapping
    else:
        from scene import Scene
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render
    points = model/'point_cloud'
    iteration = max(int(p.name.split('_')[-1]) for p in points.glob('iteration_*') if (p/'point_cloud.ply').exists())
    gaussian = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussian, load_iteration=iteration, shuffle=False)
    pipeline = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, interval=1)
    background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    extra = {}
    if job['method'] == 'dofgs':
        gaussian.setup_defocus_params(len(scene.getTrainCameras()), scene.cameras_translate, str(source))
        path = model/'defocus_params'/f'iteration_{iteration}'/'defocus_params.pth'
        if path.exists(): gaussian.load_defocus_params(str(path))
        extra['tone_map'] = ToneMapping(getattr(args, 'tonemap_type', 'none')).cuda()
    elif job['method'] == 'cocogaussian' and (model/'coco-kernel.pt').exists():
        state = torch.load(model/'coco-kernel.pt', map_location='cpu')
        saved_uids = {c['name']: c['uid'] for c in meta['train_cameras']}
        for camera in scene.getTrainCameras():
            camera.uid = saved_uids[camera.image_name]
        gaussian.create_blurring_kernel(argparse.Namespace(**state['options']), train_cams=scene.getTrainCameras())
        gaussian.CoCoBlurKernel.focal_planes = torch.nn.Parameter(state['state_dict']['focal_planes'].cuda())
        gaussian.CoCoBlurKernel.load_state_dict(state['state_dict']); gaussian.CoCoBlurKernel.max_depth = state['max_depth']
    elif job['method'] == 'dofgaussian':
        extra['lens'] = torch.load(model/'lens-parameters.pt', map_location='cuda')
    return model, source, args, gaussian, scene, pipeline, background, render, extra, meta


def render_views(job):
    import torch
    import cv2
    import numpy as np
    from torchvision.utils import save_image
    model, source, args, gaussian, scene, pipe, bg, function, extra, meta = load_model(job)
    destination = Path(job['output'])/'rendered'; destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text()) if (source/'manifest.json').exists() else None
    by_name = {Path(v['output_filename']).stem: v for v in manifest['views']} if manifest else {}
    original = Path(meta['original_source'])
    training = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
    training_indices = {c.image_name: i for i, c in enumerate(training)}
    mode_parameters = {}
    if manifest and job['method'] != 'mipsplatting':
        for mode in ['fg', 'bg']:
            members = [c for c in training if by_name[c.image_name]['selected_mode'] == mode]
            assert members, 'No training cameras for '+mode
            if job['method'] == 'dofgs':
                indices = [training_indices[c.image_name] for c in members]
                mode_parameters[mode] = {'F': float((gaussian.Fs[indices].clamp(0.01, 1000)*10).mean()),
                                         'K': float((gaussian.Ks[indices].clamp(0.01, 200)*100+100).mean())}
            elif job['method'] == 'dofgaussian':
                indices = [next(i for i, t in enumerate(scene.getTrainCameras()) if t.image_name == c.image_name) for c in members]
                mode_parameters[mode] = {'focus': float(extra['lens']['focus'][indices].mean()), 'K': float(extra['lens']['K'][indices].mean()),
                                         'effective_K': 10*float(extra['lens']['K'][indices].mean()), 'gamma': 2}
            else:
                indices = [c.uid for c in members]
                mode_parameters[mode] = {'focal_plane_normalized': float(gaussian.CoCoBlurKernel.focal_planes[indices].mean())}
                mode_parameters[mode]['focal_plane_colmap_units'] = mode_parameters[mode]['focal_plane_normalized']*gaussian.CoCoBlurKernel.max_depth
            mode_parameters[mode]['training_images'] = [c.image_name for c in members]
    def draw(camera, mode=None, own=False):
        if job['method'] == 'dofgs':
            index = training_indices.get(camera.image_name, 0)
            if own:
                return function(camera, gaussian, pipe, bg, select_view=index, **extra)
            values = mode_parameters[mode] if mode else {'F': 1., 'K': 0.}
            return function(camera, gaussian, pipe, bg, focal_distance=torch.tensor([[values['F']]], device='cuda'),
                            K_apert=torch.tensor([[values['K']]], device='cuda'), select_view=0, **extra)
        if job['method'] == 'mipsplatting':
            return function(camera, gaussian, pipe, bg, kernel_size=getattr(args, 'kernel_size', 0.1))
        if job['method'] == 'dofgaussian':
            result = function(camera, gaussian, pipe, bg)
            if mode or own:
                from bokeh_renderer.utils import render_bokeh
                if own:
                    index = next(i for i, c in enumerate(scene.getTrainCameras()) if c.image_name == camera.image_name)
                    focus, aperture = extra['lens']['focus'][index], extra['lens']['K'][index]
                else:
                    focus, aperture = mode_parameters[mode]['focus'], mode_parameters[mode]['K']
                result['render'] = render_bokeh(result['render'], result['depth'], K_bokeh=10*aperture,
                                                disp_focus=focus, gamma=2, defocus_scale=1)
            return result
        if not mode and not own:
            return function(camera, gaussian, pipe, bg, blurring=0)
        if own:
            return function(camera, gaussian, pipe, bg, blurring=1)
        saved = gaussian.CoCoBlurKernel.focal_planes[camera.uid].detach().clone()
        gaussian.CoCoBlurKernel.focal_planes[camera.uid] = mode_parameters[mode]['focal_plane_normalized']
        result = function(camera, gaussian, pipe, bg, blurring=1)
        gaussian.CoCoBlurKernel.focal_planes[camera.uid] = saved
        return result
    pairs = []; checks = []; r1_pass = True
    with torch.no_grad():
        if manifest and job['method'] != 'mipsplatting':
            for mode in ['fg', 'bg']:
                camera = next(c for c in training if by_name[c.image_name]['selected_mode'] == mode)
                image = draw(camera, own=True)['render'].clamp(0, 1)
                error = (image-camera.original_image[:3].to('cuda')).square().mean()
                checks.append({'mode': mode, 'training_image': camera.image_name, 'psnr': float(-10*torch.log10(error)), 'parameter_source': 'own learned training parameters'})
            write(destination/'r1-training-check.json', {'checks': checks, 'parameters_from_training_only': mode_parameters})
            threshold = json.loads((PROJECT/'configs/dof-gs-round1-20261105.json').read_text())['r1_training_check_min_psnr_db']
            if min(c['psnr'] for c in checks) < threshold:
                r1_pass = False
        views = [('test', c) for c in sorted(scene.getTestCameras(), key=lambda c: c.image_name)]
        if manifest and job['method'] == 'dofgs': views += [('train', c) for c in training]
        for role, camera in views:
            base = destination/'r0'/role; base.mkdir(parents=True, exist_ok=True)
            result = draw(camera); image = result['render'].clamp(0, 1)
            assert torch.isfinite(image).all(), 'Non-finite render'
            prediction = base/(camera.image_name+'.png'); save_image(image, prediction)
            native_gt = base/(camera.image_name+'-native-gt.png'); save_image(camera.original_image[:3], native_gt)
            gt_candidates = sorted((original/'images_4').glob(camera.image_name+'.*'))
            assert len(gt_candidates) == 1, gt_candidates
            pairs.append({'protocol': 'r0' if manifest else 'A', 'role': role, 'image': camera.image_name,
                          'prediction': str(prediction), 'truth': str(gt_candidates[0]), 'native_truth': str(native_gt)})
            if job['method'] == 'dofgs' and manifest:
                white = torch.ones_like(gaussian.get_xyz)
                alpha = function(camera, gaussian, pipe, torch.zeros_like(bg), focal_distance=torch.ones(1, 1, device='cuda'),
                                 K_apert=torch.zeros(1, 1, device='cuda'), override_color=white, **extra)['render'][0]
                depth = result['depth'].squeeze()/alpha.clamp_min(1e-5)
                np.save(base/(camera.image_name+'-depth.npy'), depth.cpu().numpy())
                np.save(base/(camera.image_name+'-alpha.npy'), alpha.cpu().numpy())
            if manifest and role == 'test' and job['method'] != 'mipsplatting' and r1_pass:
                for mode in ['fg', 'bg']:
                    target = destination/'r1'/mode; target.mkdir(parents=True, exist_ok=True)
                    prediction = target/(camera.image_name+'.png'); save_image(draw(camera, mode)['render'].clamp(0, 1), prediction)
                    raw = PROJECT/'data/raw/DoF-NeRF/DoF_RealWorld_Data'/job['scene']/by_name[camera.image_name]['source_files'][mode]
                    truth = target/(camera.image_name+'-gt.png')
                    assert cv2.imwrite(str(truth), cv2.resize(cv2.imread(str(raw)), (497, 331), interpolation=cv2.INTER_AREA))
                    pairs.append({'protocol': 'r1-'+mode, 'role': 'test', 'image': camera.image_name,
                                  'prediction': str(prediction), 'truth': str(truth)})
    write(destination/'pairs.json', {'method': job['method'], 'dataset': job['dataset'], 'scene': job['scene'],
                                    'model': str(model), 'pairs': pairs, 'lens_from_training': mode_parameters,
                                    'r1_training_check_passed': r1_pass,
                                    'rendering': 'self-written calls to official renderers; aperture zero for r0; DoF-Gaussian receives depth, not disparity'})


def benchmark(job):
    import torch
    import numpy as np
    config = json.loads((PROJECT/'configs/dof-gs-round1-20261105.json').read_text())
    model, source, args, gaussian, scene, pipe, bg, function, extra, meta = load_model(job)
    camera = sorted(scene.getTestCameras(), key=lambda c: c.image_name)[0]
    method = job['method']; cases = []; calibration = []
    if method == 'dofgs':
        focal = float((gaussian.Fs.clamp(0.01, 1000)*10).mean())
        aperture = float((gaussian.Ks.clamp(0.01, 200)*100+100).mean())
        def draw(value):
            return function(camera, gaussian, pipe, bg, focal_distance=torch.tensor([[focal]], device='cuda'),
                            K_apert=torch.tensor([[value]], device='cuda'), **extra)
        pinhole = draw(0)
        view_z = torch.cat([gaussian.get_xyz, torch.ones(len(gaussian.get_xyz), 1, device='cuda')], 1)@camera.world_view_transform
        visible = pinhole['visibility_filter'] & (view_z[:, 2] > 0.01)
        difference = (1/focal-1/view_z[visible, 2].clamp_min(1e-7)).abs().max().item()
        cases = [('all_in_focus', 0., 0.), ('learned_dof', aperture, .5*aperture*difference)]
        for radius in config['e4_radii_pixels']:
            value = 2*radius/max(difference, 1e-12)
            calibration.append({'target_radius_px': radius, 'parameter': value, 'actual_max_radius_px': .5*value*difference})
            cases.append(('e4_'+str(radius), value, .5*value*difference))
        del pinhole, view_z, visible
    elif method == 'dofgaussian':
        from bokeh_renderer.utils import render_bokeh
        focal = float(extra['lens']['focus'].mean()); aperture = 10*float(extra['lens']['K'].mean())
        result = function(camera, gaussian, pipe, bg)
        disparity = 1/result['depth'].clamp_min(1e-10)
        disparity = ((disparity-disparity.min())/(torch.quantile(disparity, .95)-disparity.min()).clamp_min(1e-10)).clamp(1e-5, 1.)
        difference = float((disparity-focal).abs().max())
        def draw(value):
            package = function(camera, gaussian, pipe, bg)
            if value: package['render'] = render_bokeh(package['render'], package['depth'], K_bokeh=value, disp_focus=focal, gamma=2, defocus_scale=1)
            return package
        cases = [('all_in_focus', 0., 0.), ('learned_dof', aperture, aperture*difference)]
        for radius in config['e4_radii_pixels']:
            value = radius/max(difference, 1e-12)
            calibration.append({'target_radius_px': radius, 'parameter': value, 'actual_max_radius_px': value*difference})
            cases.append(('e4_'+str(radius), value, value*difference))
        del result, disparity
    elif method == 'cocogaussian':
        kernel = gaussian.CoCoBlurKernel
        kernel.focal_planes[:] = kernel.focal_planes.mean()
        pinhole = function(camera, gaussian, pipe, bg, blurring=0)
        original = kernel.forward
        package = original(pos=gaussian.get_xyz.detach(), scales=gaussian.get_scaling.detach(), rotations=gaussian.get_rotation.detach(),
                           viewdirs=camera.camera_center.repeat(len(gaussian.get_xyz), 1), cam=camera)
        offsets = package['pos_delta'].view(-1, 3, kernel.num_coc_gaussians)
        selected = pinhole['visibility_filter']
        positions = gaussian.get_xyz[selected]; offsets = offsets[selected].permute(0, 2, 1)
        ones = torch.ones((*positions.shape[:-1], 1), device='cuda')
        base_clip = torch.cat([positions, ones], -1)@camera.full_proj_transform
        base_xy = base_clip[:, :2]/base_clip[:, 3:].clamp_min(1e-6)
        pixels = torch.tensor([camera.image_width/2, camera.image_height/2], device='cuda')
        def max_radius(factor):
            p = positions[:, None, :]+offsets*factor
            clip = torch.cat([p, torch.ones((*p.shape[:-1], 1), device='cuda')], -1)@camera.full_proj_transform
            xy = clip[..., :2]/clip[..., 3:].clamp_min(1e-6)
            return float(((xy-base_xy[:, None, :])*pixels).norm(dim=-1).max())
        def draw(value):
            if not value: return function(camera, gaussian, pipe, bg, blurring=0)
            def scaled(*a, **kw):
                output = original(*a, **kw); output['pos_delta'] = output['pos_delta']*value; return output
            kernel.forward = scaled
            try: return function(camera, gaussian, pipe, bg, blurring=1)
            finally: kernel.forward = original
        initial = max_radius(1.)
        cases = [('all_in_focus', 0., 0.), ('learned_dof', 1., initial)]
        for radius in config['e4_radii_pixels']:
            lo, hi = 0., max(1., radius/max(initial, 1e-8)*2)
            for _ in range(10):
                if max_radius(hi) >= radius: break
                hi *= 2
            for _ in range(30):
                mid = (lo+hi)/2
                if max_radius(mid) < radius: lo = mid
                else: hi = mid
            value = 0. if radius == 0 else (lo+hi)/2
            actual = max_radius(value) if value else 0.
            calibration.append({'target_radius_px': radius, 'position_offset_scale': value, 'actual_max_radius_px': actual,
                                'scope': 'self-written scaling of learned 3D position offsets; scale/rotation heads unchanged'})
            cases.append(('e4_'+str(radius), value, actual))
        del package, offsets, selected, positions, ones, base_clip, base_xy, pixels, pinhole
    else:
        def draw(value): return function(camera, gaussian, pipe, bg, kernel_size=getattr(args, 'kernel_size', .1))
        cases = [('all_in_focus', 0., 0.)]
    records = []
    for name, value, radius in cases:
        if 'result' in locals(): del result
        torch.cuda.empty_cache()
        for _ in range(config['benchmark_warmup']): draw(value)
        torch.cuda.synchronize(); baseline = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        sampler = Sampler(); sampler.start(); started = time.perf_counter()
        for _ in range(config['benchmark_repeats']): result = draw(value)
        torch.cuda.synchronize(); seconds = time.perf_counter()-started
        assert torch.isfinite(result['render']).all()
        peak = torch.cuda.max_memory_allocated(); time.sleep(.6); observed = sampler.finish()
        records.append({'case': name, 'parameter': value, 'max_radius_px': radius,
                        'milliseconds': seconds/config['benchmark_repeats']*1000, 'fps': config['benchmark_repeats']/seconds,
                        'torch_peak_bytes': peak, 'torch_incremental_peak_bytes': peak-baseline,
                        'nvidia_peak_mib': observed['gaussian']['nvidia_peak_mib'], 'repeats': config['benchmark_repeats']})
    report = {'method': method, 'scene': job['scene'], 'view': camera.image_name, 'status': 'benchmarked',
              'gaussians': len(gaussian.get_xyz), 'gaussian_parameters': 59*len(gaussian.get_xyz),
              'model_bytes': meta.get('model_bytes'), 'optimized_parameters': meta.get('optimized_parameters'),
              'results': records, 'e4_calibration': calibration, 'exclusive_gpu': True,
              'timing': 'synchronize before/after repeated renderer calls; model load, calibration, warmup and file writes excluded',
              'zero_aperture': 'pinhole rendering; scattering/additional CoCo renders bypassed',
              'radius_definition': 'DOF-GS exact CUDA circle formula; DoF-Gaussian scatter radius; CoCo max projected learned position offset over pinhole-visible Gaussians'}
    write(Path(job['benchmark_output']), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'render', 'benchmark'])
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args(); job = json.loads(args.job.read_text())
    library = str(Path(sys.prefix)/'lib')
    if os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] != library:
        cuda = '/usr/local/cuda-12.8/lib64' if job['method'] == 'dofgaussian' else str(Path(sys.prefix)/'lib')
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, LD_LIBRARY_PATH=library+':'+cuda+':'+os.environ.get('LD_LIBRARY_PATH', '')))
    guard(); subprocess.run(['nvidia-smi'], check=True)
    gpu_lock = (Path(os.environ['TMPDIR'])/'dof-gs-gpu.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX if job.get('exclusive') else fcntl.LOCK_SH)
    repo = WORKSPACE/REPOS[job['method']]; sys.path.insert(0, str(repo)); os.chdir(repo)
    if args.phase == 'train': train(job)
    elif args.phase == 'render':
        import torch
        with torch.no_grad(): render_views(job)
    else:
        import torch
        with torch.no_grad(): benchmark(job)


if __name__ == '__main__': main()
