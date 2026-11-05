#!/usr/bin/env python3
"""Run the approved COLMAP stages and each method's own split reader.

Run this script from a prepared environment with numpy. No training is performed.
The colmap phase copies raw scenes before reconstruction. The split phase uses
temporary views of each dataset because official readers may write points3D.ply.
"""
import argparse
import csv
import fcntl
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HOLDS = {'bush': 13, 'cake': 9, 'caps': 5, 'cisco': 7, 'coral': 10,
         'cupcake': 6, 'cups': 8, 'daisy': 8, 'sausage': 6, 'seal': 6, 'tools': 7}
METHODS = {'dofgs': 'DOFGS_implementation', 'dofgaussian': 'DoF-Gaussian',
           'mipsplatting': 'mip-splatting', 'cocogaussian': 'CoCoGaussian'}
SUFFIXES = {'.jpg', '.jpeg', '.png'}


def disk_check():
    subprocess.run(['df', '-h', '/', '/root/autodl-tmp'], check=True)
    if shutil.disk_usage('/').free < 1024**3:
        raise RuntimeError('STOP: system disk below 1 GiB')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def image_files(path):
    return sorted(p for p in path.iterdir() if p.suffix.lower() in SUFFIXES)


def model_images(path):
    with (path/'images.bin').open('rb') as stream:
        return struct.unpack('<Q', stream.read(8))[0]


def model_reader(workspace):
    path = workspace/'DoF-Gaussian/utils/colmap_read_model.py'
    spec = importlib.util.spec_from_file_location('repro_colmap_model_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_poses_bounds(scene, workspace):
    """Standard LLFF conversion of newly estimated geometry for CoCo's loader."""
    import numpy as np
    module = model_reader(workspace)
    cameras, images, points = module.read_model(str(scene/'sparse/0'), ext='.bin')
    rows = []
    for image in sorted(images.values(), key=lambda item: item.name):
        camera = cameras[image.camera_id]
        rotation = image.qvec2rotmat()
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = image.tvec
        c2w = np.linalg.inv(world_to_camera)[:3, :4]
        # COLMAP right/down/forward -> LLFF down/right/back.
        pose = np.concatenate([c2w[:, 1:2], c2w[:, 0:1], -c2w[:, 2:3], c2w[:, 3:4],
                               np.array([[camera.height], [camera.width], [camera.params[0]]])], axis=1)
        depths = [(rotation@points[int(pid)].xyz + image.tvec)[2]
                  for pid in image.point3D_ids if pid != -1 and int(pid) in points]
        depths = np.asarray([z for z in depths if z > 0])
        if not len(depths):
            raise ValueError('No positive sparse depths for '+image.name)
        near, far = np.percentile(depths, [0.1, 99.9])
        rows.append(np.r_[pose.ravel(), near, far])
    np.save(scene/'poses_bounds.npy', np.stack(rows))


def copy_scene(source, destination, preserve_sparse=False):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'images_4', 'poses_bounds.npy', 'manifest.json', 'rectification.json', 'rectification-maps'):
        item = source/name
        target = destination/name
        if not item.exists() or target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    for item in source.glob('hold=*'):
        if not (destination/item.name).exists():
            shutil.copy2(item, destination/item.name)
    sparse_target = destination/('sparse_deblurnerf' if preserve_sparse else 'sparse')
    if not sparse_target.exists() and (source/'sparse').is_dir():
        shutil.copytree(source/'sparse', sparse_target)


def execute(colmap, command, arguments, log):
    disk_check()
    pending = Path(os.environ['TMPDIR'])/'dof-gs-round1-exclusive.pending'
    overlap = os.environ.get('GS_COLMAP_TRAINING_OVERLAP')=='1'
    while pending.exists() and not overlap: time.sleep(5)
    gpu_lock = None
    if command in {'feature_extractor', 'exhaustive_matcher', 'patch_match_stereo'}:
        gpu_lock = (Path(os.environ['TMPDIR'])/'dof-gs-gpu.lock').open('a')
        fcntl.flock(gpu_lock, fcntl.LOCK_SH if overlap else fcntl.LOCK_EX)
        while pending.exists() and not overlap:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN); time.sleep(5); fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        subprocess.run(['nvidia-smi'], check=True)
    start = time.monotonic()
    with log.open('a') as stream:
        argv = [str(colmap), command] + list(map(str, arguments))
        stream.write('\nCOMMAND '+json.dumps(argv)+'\n'); stream.flush()
        try:
            if gpu_lock:
                wrapper=Path(__file__).with_name('gs_round1_resources.py')
                subprocess.run([sys.executable,str(wrapper),'4096','colmap:'+command,*argv],
                               stdout=stream,stderr=subprocess.STDOUT,
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),check=True)
            else:
                subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'), check=True)
        finally:
            if gpu_lock: gpu_lock.close()
    disk_check()
    return time.monotonic()-start


def sparse(scene, colmap, log, cpu_sift=False):
    elapsed = {}
    database = scene/'db.db'
    if not (scene/'sparse/0/images.bin').exists():
        elapsed['feature_extractor'] = execute(colmap, 'feature_extractor',
            ['--database_path', database, '--image_path', scene/'images', '--ImageReader.camera_model', 'SIMPLE_RADIAL']
            + (['--SiftExtraction.use_gpu', '0'] if cpu_sift else []), log)
        elapsed['exhaustive_matcher'] = execute(colmap, 'exhaustive_matcher',
            ['--database_path', database, '--SiftMatching.guided_matching', '1']
            + (['--SiftMatching.use_gpu', '0'] if cpu_sift else []), log)
        (scene/'sparse').mkdir(exist_ok=True)
        elapsed['mapper'] = execute(colmap, 'mapper',
            ['--database_path', database, '--image_path', scene/'images', '--output_path', scene/'sparse'], log)
    total = len(image_files(scene/'images'))
    registered = model_images(scene/'sparse/0')
    return {'registered_images': registered, 'total_images': total, 'all_registered': registered == total,
            'seconds': elapsed, 'model': str(scene/'sparse/0')}


def dense(scene, colmap, log):
    elapsed = {}
    target = scene/'dense'
    target.mkdir(exist_ok=True)
    if not (target/'sparse/images.bin').exists():
        elapsed['image_undistorter'] = execute(colmap, 'image_undistorter',
            ['--image_path', scene/'images', '--input_path', scene/'sparse/0', '--output_path', target,
             '--output_type', 'COLMAP'], log)
    expected = [p.name+'.geometric.bin' for p in image_files(target/'images')]
    if not all((target/'stereo/depth_maps'/name).exists() for name in expected):
        elapsed['patch_match_stereo'] = execute(colmap, 'patch_match_stereo',
            ['--workspace_path', target, '--workspace_format', 'COLMAP'], log)
    if not (target/'fused.ply').exists() or not (target/'fused.ply.vis').exists():
        elapsed['stereo_fusion'] = execute(colmap, 'stereo_fusion',
            ['--workspace_path', target, '--workspace_format', 'COLMAP', '--input_type', 'geometric',
             '--output_path', target/'fused.ply', '--StereoFusion.min_num_pixels', '2'], log)
    missing = [name for name in expected if not (target/'stereo/depth_maps'/name).exists()]
    with (target/'fused.ply.vis').open('rb') as stream:
        count = struct.unpack('<Q', stream.read(8))[0]
    return {'seconds': elapsed, 'fused_ply': str(target/'fused.ply'),
            'visibility': str(target/'fused.ply.vis'), 'fused_points': count,
            'expected_depth_maps': len(expected), 'missing_geometric_depth_maps': missing,
            'complete': not missing and count > 0}


def colmap_phase(args):
    raw, processed = args.project/'data/raw', args.project/'data/processed'
    report = json.loads(args.record.read_text()) if args.record.exists() else {'runs': {}}
    report['executable'] = str(args.colmap)
    report['protocol'] = 'DoF-Gaussian colmap.sh; SIMPLE_RADIAL; guided exhaustive matching; fusion min_num_pixels=2'
    args.logs.mkdir(parents=True, exist_ok=True)
    available = args.colmap.is_file()
    if available:
        available = subprocess.run([str(args.colmap), '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    for scene_name in ([x for x in HOLDS if x != 'bush' and (not args.scenes or x in args.scenes)] if args.dataset != 'dofnerf' else []):
        key = 'deblurnerf/'+scene_name
        if report['runs'].get(key, {}).get('status') == 'complete':
            continue
        source = raw/'DeblurNeRF/real_defocus_blur'/('defocus'+scene_name)
        target = processed/'deblurnerf-dofgaussian'/source.name
        row = {'source': str(source), 'output': str(target)}
        try:
            expected_count = {'cake': 34, 'caps': 34, 'cisco': 45, 'coral': 49, 'cupcake': 44,
                              'cups': 43, 'daisy': 52, 'sausage': 39, 'seal': 39, 'tools': 41}[scene_name]
            if len(image_files(source/'images')) != expected_count:
                raise ValueError('Raw image count does not match the approved dataset')
            copy_scene(source, target, preserve_sparse=True)
            if not available:
                raise FileNotFoundError('Modified COLMAP executable unavailable')
            row['sparse'] = sparse(target, args.colmap, args.logs/('colmap-'+scene_name+'.log'))
            row['dense'] = dense(target, args.colmap, args.logs/('colmap-'+scene_name+'.log'))
            row['status'] = 'complete' if row['sparse']['all_registered'] and row['dense']['complete'] else 'incomplete'
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            row['status'] = 'blocked'; row['error'] = str(exc)
        report['runs'][key] = row; save(args.record, report)
    scenes_root = processed/('dofnerf-scenes-provided' if (args.logs/'provided-poses.finished').exists() else 'dofnerf-scenes')
    for name in ([x for x in ('amiya', 'camera', 'plant', 'kendo', 'desk', 'shelf', 'turtle') if not args.scenes or x in args.scenes] if args.dataset != 'deblurnerf' else []):
        key = 'dofnerf/sparse/'+name
        row = report['runs'].get(key, {})
        mix = scenes_root/'mix'/name
        try:
            if not (mix/'manifest.json').exists():
                raise FileNotFoundError('DoF-NeRF scene was not prepared: '+name)
            sparse_colmap = args.colmap if available else args.sparse_colmap
            if not sparse_colmap or not sparse_colmap.is_file():
                raise FileNotFoundError('Modified COLMAP unavailable; sparse fallback must be explicitly recorded')
            provided=(mix/'rectification.json').exists()
            row = sparse(mix, sparse_colmap, args.logs/('colmap-dofnerf-'+name+'.log'), cpu_sift=not available)
            row['colmap_executable'] = str(sparse_colmap)
            row['fallback_unmodified_colmap'] = not available
            if not row['all_registered']:
                raise ValueError('Registration incomplete; all-sharp pose estimation is not authorized')
            if not provided:write_poses_bounds(mix, args.workspace)
            import filecmp
            for layout in ('fg', 'bg'):
                target = scenes_root/layout/name
                if not (target/'sparse').exists():
                    shutil.copytree(mix/'sparse', target/'sparse')
                if not (target/'poses_bounds.npy').exists():
                    shutil.copy2(mix/'poses_bounds.npy', target/'poses_bounds.npy')
                if not filecmp.cmp(mix/'sparse/0/images.bin', target/'sparse/0/images.bin', shallow=False):
                    raise ValueError('Copied camera poses differ: '+str(target))
            row['geometry_identical_across_layouts'] = True
            row['poses_bounds'] = 'provided geometry renamed/scaled/rectified, world poses unchanged' if provided else 'standard LLFF conversion from estimated mix sparse model; no provided geometry used'
            row['status'] = 'complete'
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            row['status'] = 'blocked'; row['error'] = str(exc)
        report['runs'][key] = row; save(args.record, report)
        for layout in ('mix', 'fg', 'bg'):
            dense_key = f'dofnerf/dense/{layout}/{name}'
            if report['runs'].get(dense_key,{}).get('status')=='complete':continue
            if row['status'] != 'complete' or not available:
                report['runs'][dense_key] = {'status': 'blocked', 'error': 'Sparse stage blocked or modified COLMAP unavailable'}
                continue
            target = processed/('dofnerf-dofgaussian-provided' if scenes_root.name.endswith('provided') else 'dofnerf-dofgaussian')/layout/name
            dense_row = {'output': str(target), 'fixed_geometry_source': str(mix/'sparse/0')}
            try:
                copy_scene(scenes_root/layout/name, target)
                dense_row['sparse'] = {'registered_images': model_images(target/'sparse/0'),
                                       'total_images': len(image_files(target/'images'))}
                dense_row['dense'] = dense(target, args.colmap, args.logs/(f'colmap-dofnerf-{layout}-{name}.log'))
                dense_row['status'] = 'complete' if dense_row['dense']['complete'] else 'incomplete'
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                dense_row['status'] = 'blocked'; dense_row['error'] = str(exc)
            report['runs'][dense_key] = dense_row; save(args.record, report)
        save(args.record, report)


def split_worker(args):
    """Call an unmodified official reader on a writable temporary view."""
    sys.path.insert(0, str(args.workspace/METHODS[args.method]))
    os.chdir(args.workspace/METHODS[args.method])
    result = {'method': args.method, 'source': str(args.source), 'hold': args.hold}
    try:
        if args.method == 'dofgs':
            reader = importlib.import_module('src.scene.dataset_readers')
            import yaml
            config = args.workspace/METHODS[args.method]/'config_files'/('scene_info_dofnerf.yaml' if args.hold == 9 and args.source.name in {'amiya','camera','plant','kendo','desk','shelf','turtle'} else 'scene_info.yaml')
            scenes = yaml.safe_load(config.read_text())
            candidates = [value for value in scenes.values() if Path(value['source_path']).name == args.source.name]
            if candidates:
                assert all(int(value['holdout']) == args.hold for value in candidates)
            result['scene_config'] = str(config)
        else:
            reader = importlib.import_module('scene.dataset_readers')
        with tempfile.TemporaryDirectory(prefix='gs-split-', dir=os.environ['TMPDIR']) as tmp:
            view = Path(tmp)/('synthetic-view' if args.synthetic else 'scene')
            view.mkdir()
            for name in ('images', 'images_4', 'input', 'poses_bounds.npy', 'dense'):
                if (args.source/name).exists():
                    (view/name).symlink_to((args.source/name).resolve(), target_is_directory=(args.source/name).is_dir())
            for marker in args.source.glob('hold=*'):
                (view/marker.name).touch()
            shutil.copytree(args.source/'sparse', view/'sparse')
            hold = args.hold
            if args.method == 'cocogaussian':
                hold = 8
                for name in os.listdir(view):
                    if name.startswith('hold='):
                        hold = int(name.split('=')[-1])
                result['hold_from_marker'] = hold
            info = reader.readColmapSceneInfo(str(view), 'images', True, llffhold=hold)
            train = sorted(Path(c.image_path).name for c in info.train_cameras)
            test = sorted(Path(c.image_path).name for c in info.test_cameras)
            folder = 'input' if args.method == 'dofgs' and args.synthetic else 'images'
            expected_files = [p.name for p in image_files(args.source/folder)]
            expected_test = expected_files[::args.hold]
            result.update(train_files=train, test_files=test, expected_test_files=expected_test,
                          train_count=len(train), test_count=len(test),
                          matches=test == expected_test and sorted(train+test) == expected_files,
                          status='checked')
            # FOV stays fixed when pixels and focal length are scaled together.
            actual_size = lambda image: (int(image.shape[1]), int(image.shape[0])) if hasattr(image, 'shape') else image.size
            result['camera_image_sizes'] = sorted({(int(c.width), int(c.height), *actual_size(c.image))
                                                   for c in info.train_cameras+info.test_cameras})
            result['resolution_4_sizes'] = sorted({(round(actual_size(c.image)[0]/4), round(actual_size(c.image)[1]/4))
                                                  for c in info.train_cameras+info.test_cameras})
    except Exception as exc:
        result.update(status='blocked', error=f'{type(exc).__name__}: {exc}', matches=None,
                      train_count=None, test_count=None)
    save(args.record, result)


def split_phase(args):
    raw, processed = args.project/'data/raw', args.project/'data/processed'
    targets = []
    for name, hold in HOLDS.items():
        if name == 'bush':
            continue
        source = raw/'DeblurNeRF/real_defocus_blur'/('defocus'+name)
        for method in ('dofgs', 'mipsplatting'):
            targets.append((method, 'deblurnerf-real', name, source, hold, False))
        targets.append(('dofgaussian', 'deblurnerf-real', name,
                        processed/'deblurnerf-dofgaussian'/source.name, hold, False))
        candidates = [p for p in (raw/'Deblurring3DGS').rglob('defocus'+name) if (p/'images').is_dir()]
        targets.append(('cocogaussian', 'deblurring3dgs-real', name,
                        candidates[0] if len(candidates) == 1 else raw/'Deblurring3DGS'/('defocus'+name), hold, False))
    for name in ('cozy2room', 'factory', 'pool', 'tanabata', 'wine'):
        source = raw/'DOFGS-author/synthetic_defocus_blur'/('defocus'+name)
        candidates = [p for p in (raw/'DOFGS-author/synthetic_defocus_blur').rglob('defocus'+name) if (p/'sparse').is_dir()]
        if len(candidates) == 1:
            source = candidates[0]
        targets.append(('dofgs', 'dofgs-author-synthetic', name, source, 8, True))
    for layout in ('mix', 'fg', 'bg'):
        for name in ('amiya', 'camera', 'plant', 'kendo', 'desk', 'shelf', 'turtle'):
            for method in METHODS:
                provided=(processed/'dofnerf-scenes-provided'/layout/name/'rectification.json').exists()
                directory = ('dofnerf-dofgaussian-provided' if provided else 'dofnerf-dofgaussian') if method == 'dofgaussian' else ('dofnerf-scenes-provided' if provided else 'dofnerf-scenes')
                targets.append((method, 'dofnerf-'+layout, name, processed/directory/layout/name, 9, False))
    previous = json.loads(args.record.with_suffix('.json').read_text()).get('checks', []) if args.record.with_suffix('.json').exists() else []
    if args.dataset!='all':
        targets=[t for t in targets if t[1].startswith('dofnerf')== (args.dataset=='dofnerf')]
    if args.scenes:targets=[t for t in targets if t[2] in args.scenes]
    if args.layouts:targets=[t for t in targets if not t[1].startswith('dofnerf') or t[1].split('-')[-1] in args.layouts]
    cache = {(r['method'], r['dataset'], r['scene']): r for r in previous if r.get('matches') is True}
    selected={(m,d,n) for m,d,n,*_ in targets}
    details = [r for r in previous if (r['method'],r['dataset'],r['scene']) not in selected]
    for method, dataset, name, source, hold, synthetic in targets:
        if (method, dataset, name) in cache and cache[(method,dataset,name)].get('source')==str(source):
            details.append(cache[(method, dataset, name)]); continue
        disk_check()
        output = args.logs/(f'split-{method}-{dataset}-{name}.json')
        python = args.workspace/'envs'/method/'bin/python'
        if not python.exists() or not (source/'sparse/0/images.bin').exists():
            row = {'method': method, 'dataset': dataset, 'scene': name, 'source': str(source), 'hold': hold,
                   'status': 'blocked', 'error': 'Environment or sparse model unavailable', 'matches': None,
                   'train_count': None, 'test_count': None}
        else:
            command = [str(python), str(Path(__file__).resolve()), 'split-worker', '--method', method,
                       '--workspace', str(args.workspace), '--source', str(source.resolve()),
                       '--hold', str(hold), '--record', str(output)]
            if synthetic:
                command.append('--synthetic')
            with output.with_suffix('.log').open('w') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES='0')).returncode
            row = json.loads(output.read_text()) if code == 0 and output.exists() else {
                'method': method, 'status': 'blocked', 'error': f'Reader process exit={code}',
                'matches': None, 'train_count': None, 'test_count': None}
            row.update(dataset=dataset, scene=name)
        details.append(row)
    save(args.record.with_suffix('.json'), {'checks': details})
    fields = ['method', 'dataset', 'scene', 'hold', 'train_count', 'test_count', 'matches', 'status', 'error']
    with args.record.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader(); writer.writerows(details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['colmap', 'splits', 'split-worker'])
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--workspace', type=Path, default=Path('/root/autodl-tmp'))
    parser.add_argument('--colmap', type=Path, default=Path('/root/autodl-tmp/tools/colmap-point-visibility/install/bin/colmap'))
    parser.add_argument('--sparse-colmap', type=Path, help='Explicitly recorded unmodified COLMAP fallback, sparse only')
    parser.add_argument('--logs', type=Path, default=Path('/root/autodl-tmp/OPPO-depth-guided-video-bokeh/logs/dof-gs-repro-2026-11-05'))
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--hold', type=int)
    parser.add_argument('--method', choices=list(METHODS))
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--dataset', choices=['all', 'deblurnerf', 'dofnerf'], default='all')
    parser.add_argument('--scenes', nargs='*')
    parser.add_argument('--layouts', nargs='*',choices=['mix','fg','bg'])
    args = parser.parse_args()
    if args.phase == 'colmap':
        with (Path(os.environ['TMPDIR'])/'dof-gs-colmap-pipeline.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            colmap_phase(args)
    elif args.phase == 'splits':
        with (args.logs/'split-record.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            split_phase(args)
    else:
        split_worker(args)


if __name__ == '__main__':
    main()
