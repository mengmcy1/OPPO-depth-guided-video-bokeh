"""Explicit translated/off-axis pinhole jobs for U0 and separated U1 views.

Run with the fixed portable Blender. No original scene file is written.
Each job names its camera offset in original OpenCV camera coordinates.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    config = json.loads(args.jobs.read_text())
    out = Path(config['output']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = out / 'timings.json'
    rows = json.loads(summary.read_text()) if summary.exists() else []
    done = {r['key'] for r in rows}
    bpy.ops.wm.open_mainfile(filepath=str(Path(config['blend']).resolve()))
    scene = bpy.context.scene
    original = scene.camera
    camera = original.copy()
    camera.data = original.data.copy()
    camera.animation_data_clear()
    camera.data.animation_data_clear()
    camera.parent = None
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 8
    scene.render.use_motion_blur = False
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    camera.data.dof.use_dof = False
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.clip_start = .0001
    camera.data.clip_end = 100000.
    camera.data.lens = 50.
    camera.data.sensor_width = 36.
    if not args.geometry_only:
        scene.cycles.device = 'GPU'
        pref = bpy.context.preferences.addons['cycles'].preferences
        pref.compute_device_type = 'OPTIX'
        pref.get_devices()
        for device in pref.devices:
            device.use = device.type == 'OPTIX'
        if not any(d.use for d in pref.devices):
            raise RuntimeError('No enabled OptiX GPU')
    checks = []
    for job in config['jobs']:
        if (out / 'STOP-RENDER').exists():
            break
        key = job['key']
        if key in done:
            continue
        scene.frame_set(job['frame'])
        base = np.array(original.matrix_world, dtype=np.float64)
        cv = base @ np.diag([1., -1., -1., 1.])
        xy = np.array(job['xy_m'], np.float64)
        matrix = base.copy()
        matrix[:3, 3] += cv[:3, :3] @ np.r_[xy, 0.]
        camera.matrix_world = Matrix(matrix.tolist())
        K = np.array(config['K'], np.float64)
        intr = K.copy()
        focus = job.get('focus_m')
        if focus is not None:
            intr[:2, 2] += np.diag(K)[:2] * xy / focus
        camera.data.shift_x = -(intr[0, 2] - 256.) / 512.
        camera.data.shift_y = (intr[1, 2] - 256.) / 512.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        P = np.array(camera.calc_matrix_camera(depsgraph, x=512, y=512), np.float64)
        # Actual Blender projection against the exact OpenCV off-axis camera.
        pts = np.array([[x, y, z] for z in [.8, 1.4, 5., 10.]
                        for x, y in [(-.2, -.1), (0., 0.), (.2, .1)]])
        clip = np.c_[pts * [1., -1., -1.], np.ones(len(pts))] @ P.T
        ndc = clip[:, :2] / clip[:, 3:]
        uv_blender = np.c_[(ndc[:, 0] + 1) * 256, (1 - ndc[:, 1]) * 256]
        uv_cv = (pts @ intr.T)[:, :2] / pts[:, 2:]
        err = float(abs(uv_cv - uv_blender).max())
        if err > .05:
            raise RuntimeError(f'Projection mismatch {err}px for {key}')
        focal_error = None
        if focus is not None:
            central = np.array([.07, -.03, focus])
            moved = central - np.r_[xy, 0.]
            a, b = K @ central, intr @ moved
            focal_error = float(abs(a[:2] / a[2] - b[:2] / b[2]).max())
            if focal_error > 1e-7:
                raise RuntimeError('Focal plane mismatch')
        pose = cv.copy()
        pose[:3, 3] = matrix[:3, 3]
        check = dict(key=key, frame=job['frame'], xy_m=xy.tolist(),
                     focus_m=focus, K=intr.tolist(), camera_c2w=pose.tolist(),
                     projection_error_px=err, focal_plane_error_px=focal_error)
        checks.append(check)
        if args.geometry_only:
            continue
        path = out / (key + '.exr')
        if path.exists():
            raise FileExistsError(path)
        scene.cycles.samples = job.get('spp', 1024)
        scene.cycles.seed = job.get('seed', 2026111102)
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = str(path)
        start = time.monotonic()
        bpy.ops.render.render(write_still=True)
        row = dict(**check, spp=scene.cycles.samples, seed=scene.cycles.seed,
                   seconds=time.monotonic() - start, path=str(path), bytes=path.stat().st_size,
                   role=job['role'], fstop=job.get('fstop'), sample=job.get('sample'))
        rows.append(row)
        summary.write_text(json.dumps(rows, indent=2) + '\n')
        print('UPPER_PINHOLE', key, row['seconds'], flush=True)
    (out / ('geometry-checks.json' if args.geometry_only else 'projection-checks.json')).write_text(
        json.dumps(dict(completed=True, blender=bpy.app.version_string,
                        main_scene_modified=False, rows=checks), indent=2) + '\n')


if __name__ == '__main__':
    main()
