"""Render the C audit's matched B-tier optical configuration in Blender."""
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix


def render(out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 1024
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = 410782
    scene.cycles.pixel_filter_type = 'BOX'
    scene.render.filter_size = 1.0
    scene.render.use_motion_blur = False
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 8
    size = 128
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '32'
    scene.view_settings.view_transform = 'Raw'
    scene.world = bpy.data.worlds.new('Black')
    scene.world.color = (0, 0, 0)
    cam = bpy.data.cameras.new('CV camera')
    obj = bpy.data.objects.new('CV camera', cam)
    scene.collection.objects.link(obj)
    obj.matrix_world = Matrix.Diagonal((1., -1., -1., 1.))
    scene.camera = obj
    cam.lens = 50.
    cam.sensor_width = 36.
    cam.sensor_fit = 'HORIZONTAL'
    cam.dof.use_dof = True
    cam.dof.aperture_fstop = 1.4
    cam.dof.aperture_blades = 0
    planes = [{'rectangle': [1.4, -.33, .028, -.52, .29], 'color': [.15, .33, .67]},
              {'rectangle': [2.7, -.05, .52, -.62, .38], 'color': [.74, .18, .09]},
              {'rectangle': [5., -10., 10., -10., 10.], 'color': [.55, .79, .22]}]
    for j, plane in enumerate(planes):
        z, l, r, t, b = plane['rectangle']
        verts = [(l, t, z), (r, t, z), (r, b, z), (l, b, z)]
        mesh = bpy.data.meshes.new(f'plane-{j}')
        mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
        card = bpy.data.objects.new(f'plane-{j}', mesh)
        scene.collection.objects.link(card)
        mat = bpy.data.materials.new(f'emission-{j}')
        mat.use_nodes = True
        output = mat.node_tree.nodes.get('Material Output')
        em = mat.node_tree.nodes.new('ShaderNodeEmission')
        em.inputs['Color'].default_value = (*plane['color'], 1)
        mat.node_tree.links.new(em.outputs['Emission'], output.inputs['Surface'])
        mesh.materials.append(mat)
    started = time.monotonic()
    for focus in (1.4, 5.):
        cam.dof.focus_distance = focus
        scene.render.filepath = str(out / f'F{focus:g}.exr')
        bpy.ops.render.render(write_still=True)
    fx = size * 50 / 36
    spec = dict(planes=planes, K=[[fx, 0, size/2], [0, fx, size/2], [0, 0, 1]],
                size=size, focus_m=[1.4, 5.0], fstop=1.4, blender=bpy.app.version_string,
                cycles_samples=1024, seconds=time.monotonic()-started, filter='BOX 1.0',
                exr='scene-linear float32', seed=scene.cycles.seed, denoising=False, motion_blur=False)
    (out / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')


if __name__ == '__main__':
    render(Path(sys.argv[sys.argv.index('--')+1]).resolve())
