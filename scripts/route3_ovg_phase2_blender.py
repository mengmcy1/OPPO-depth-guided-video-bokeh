"""Cycles foreground-only pass across a per-camera-pixel axial depth cut."""
import json
from pathlib import Path
import shutil
import sys
import time

import bpy
import numpy as np


def gate_materials(scene, cutter):
    """Hide the complementary first-hit radiance only on camera rays."""
    controls = []
    materials = {slot.material for obj in scene.objects if obj.type == 'MESH'
                 for slot in obj.material_slots if slot.material is not None}
    for material in materials:
        if not material.use_nodes:
            raise ValueError(f'Material requires a Cycles surface node tree: {material.name}')
        nodes, links = material.node_tree.nodes, material.node_tree.links
        output = next(n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)
        surface = output.inputs['Surface']
        original = surface.links[0].from_socket
        camera = nodes.new('ShaderNodeCameraData')
        light = nodes.new('ShaderNodeLightPath')
        uv = nodes.new('ShaderNodeTexCoord')
        image = nodes.new('ShaderNodeTexImage')
        image.image = cutter
        image.interpolation = 'Closest'
        image.extension = 'EXTEND'
        links.new(uv.outputs['Window'], image.inputs['Vector'])
        behind = nodes.new('ShaderNodeMath')
        behind.operation = 'GREATER_THAN'
        links.new(camera.outputs['View Z Depth'], behind.inputs[0])
        links.new(image.outputs['Color'], behind.inputs[1])
        invert = nodes.new('ShaderNodeMath')
        invert.operation = 'ABSOLUTE'
        which = nodes.new('ShaderNodeMath')
        which.operation = 'SUBTRACT'
        links.new(behind.outputs[0], which.inputs[0])
        links.new(which.outputs[0], invert.inputs[0])
        camera_only = nodes.new('ShaderNodeMath')
        camera_only.operation = 'MULTIPLY'
        links.new(light.outputs['Is Camera Ray'], camera_only.inputs[0])
        links.new(invert.outputs[0], camera_only.inputs[1])
        mix = nodes.new('ShaderNodeMixShader')
        links.new(camera_only.outputs[0], mix.inputs[0])
        links.new(original, mix.inputs[1])
        links.new(nodes.new('ShaderNodeHoldout').outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], surface)
        controls.append(which.inputs[1])
    if not controls:
        raise ValueError('Loaded scene has no depth-gated surfaces')
    return controls


def run(jobs_file):
    jobs_file = Path(jobs_file)
    jobs = json.loads(jobs_file.read_text())
    log = jobs_file.with_suffix('.results.json')
    rows = json.loads(log.read_text()) if log.exists() else []
    done = {row['id'] for row in rows}
    current = None
    for job in jobs:
        if job['id'] in done:
            continue
        output = Path(job['output'])
        if shutil.disk_usage(output.parent).free < 60*2**30:
            raise RuntimeError('Data disk below 60 GiB')
        if current != job['blend']:
            bpy.ops.wm.open_mainfile(filepath=job['blend'])
            current = job['blend']
            scene = bpy.context.scene
            camera = scene.camera
            cutter = bpy.data.images.new('label_cutoff_axial', 512, 512, alpha=True, float_buffer=True)
            cutter.colorspace_settings.name = 'Non-Color'
            cutter.alpha_mode = 'CHANNEL_PACKED'
            controls = gate_materials(scene, cutter)
            scene.render.engine = 'CYCLES'
            scene.cycles.device = 'GPU'
            pref = bpy.context.preferences.addons['cycles'].preferences
            pref.compute_device_type = 'OPTIX'
            pref.get_devices()
            for device in pref.devices:
                device.use = device.type == 'OPTIX'
            if not any(d.use for d in pref.devices):
                raise RuntimeError('OptiX GPU missing')
            scene.render.resolution_x = scene.render.resolution_y = 512
            scene.render.resolution_percentage = 100
            scene.render.film_transparent = True
            scene.render.use_compositing = False
            scene.render.use_motion_blur = False
            scene.cycles.use_denoising = False
            scene.cycles.use_adaptive_sampling = False
            scene.cycles.pixel_filter_type = 'BOX'
            scene.cycles.filter_width = 1.
            settings = scene.render.image_settings
            settings.file_format = 'OPEN_EXR'
            settings.color_mode = 'RGBA'
            settings.color_depth = '32'
            settings.exr_codec = 'ZIP'
        scene.frame_set(job['frame'])
        camera.data.dof.use_dof = False
        with np.load(job['coverage']) as data:
            K = data['K']
            pose = data['camera_c2w'] @ np.diag([1, -1, -1, 1])
            cutoff = data['cutoff_m']
        if not np.allclose(np.array(camera.matrix_world), pose, atol=2e-6, rtol=0):
            raise ValueError(f'Camera pose differs from exported mesh camera: {job["id"]}')
        focal = 512*camera.data.lens/camera.data.sensor_width
        if not np.allclose(K, [[focal, 0, 256],[0, focal, 256],[0, 0, 1]], atol=1e-4):
            raise ValueError(f'Intrinsics differ: {job["id"]}')
        values = np.ones((512, 512, 4), np.float32)
        values[..., :3] = cutoff[::-1, :, None]
        cutter.pixels.foreach_set(values.ravel())
        cutter.update()
        for control in controls:
            control.default_value = float(job['pass'] == 'visible-background')
        scene.cycles.seed = job['seed']
        scene.cycles.samples = job['spp']
        temporary = output.with_name(output.stem+'-writing.exr')
        scene.render.filepath = str(temporary)
        start = time.monotonic()
        bpy.ops.render.render(write_still=True)
        temporary.replace(output)
        rows.append(dict(job, seconds=time.monotonic()-start, bytes=output.stat().st_size,
                         blender=bpy.app.version_string, filter='BOX 1.0'))
        tmp = log.with_suffix('.part')
        tmp.write_text(json.dumps(rows, indent=2)+'\n')
        tmp.replace(log)
        print('PHASE2_RENDER',job['id'],rows[-1]['seconds'],flush=True)


if __name__=='__main__':
    run(sys.argv[sys.argv.index('--')+1])
