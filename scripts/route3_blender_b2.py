"""Blender4.5.3 B2: preserve low, construct high, export rigid geometry and sparse Cycles images.

Only full pinhole sequences are allowed; DOF renders are explicitly sparse.
All materials and meshes are procedural, with no downloaded assets.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Vector

INPUTS=[1,6,11,16,21,26,30,35,40,45,50,55,60]
FOREGROUND=['static_foreground_left','static_foreground_right','static_foreground_crossbar','rigid_moving_sphere']


def natural_material(name):
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;l=m.node_tree.links
    bs=n.get('Principled BSDF');bs.inputs['Roughness'].default_value=.55
    coord=n.new('ShaderNodeTexCoord');noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=7.3
    noise.inputs['Detail'].default_value=5.;noise.inputs['Roughness'].default_value=.72
    l.new(coord.outputs['Generated'],noise.inputs['Vector'])
    fine=n.new('ShaderNodeTexNoise');fine.inputs['Scale'].default_value=89.7;fine.inputs['Detail'].default_value=3.2
    l.new(coord.outputs['Generated'],fine.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(.035,.075,.09,1);ramp.color_ramp.elements[1].color=(.45,.65,.55,1)
    l.new(noise.outputs['Fac'],ramp.inputs[0]);mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.45
    l.new(ramp.outputs['Color'],mix.inputs[1]);l.new(fine.outputs['Color'],mix.inputs[2]);l.new(mix.outputs[0],bs.inputs['Base Color'])
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.15;bump.inputs['Distance'].default_value=.02
    l.new(fine.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs['Normal'],bs.inputs['Normal'])
    return m


def set_box(obj,location,dimensions):
    obj.location=location;obj.dimensions=dimensions


def high_scene(scene):
    objects=bpy.data.objects
    set_box(objects['hidden_checker_background'],(0,3.4,1.2),(4.,.1,2.4))
    set_box(objects['static_foreground_left'],(-.32,-.1,1.0),(.06,.1,1.0))
    set_box(objects['static_foreground_right'],(.35,0.,.95),(.06,.1,1.1))
    set_box(objects['static_foreground_crossbar'],(0,-.08,.8),(.7,.1,.06))
    objects['floor'].data.materials.clear();objects['floor'].data.materials.append(natural_material('natural_floor'))
    checker=objects['hidden_checker_background'].data.materials[0]
    n=checker.node_tree.nodes;l=checker.node_tree.links;bs=n.get('Principled BSDF')
    original=next(x for x in n if x.type=='TEX_CHECKER');noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=54.3;noise.inputs['Detail'].default_value=4.5
    coord=next(x for x in n if x.type=='TEX_COORD');l.new(coord.outputs['Generated'],noise.inputs['Vector'])
    mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.55;l.new(original.outputs['Color'],mix.inputs[1]);l.new(noise.outputs['Color'],mix.inputs[2]);l.new(mix.outputs[0],bs.inputs['Base Color'])
    moving=objects['rigid_moving_sphere'];moving.animation_data_clear()
    for vertex in moving.data.vertices:vertex.co*=.5
    moving.data.materials.clear();moving.data.materials.append(natural_material('natural_moving_sphere'))
    camera=scene.camera;camera.animation_data_clear()
    for frame in range(1,61):
        t=(frame-1)/59
        moving.location=(-.48+.96*t,.2+.04*math.sin(2*math.pi*t),1.05+.055*math.sin(4*math.pi*t))
        moving.rotation_euler=(.4*t,.8*t,2.5*t);moving.keyframe_insert(data_path='location',frame=frame);moving.keyframe_insert(data_path='rotation_euler',frame=frame)
        camera.location=(-.065+.13*t+.012*math.sin(13*math.pi*t),-1.5+.02*math.sin(3*math.pi*t),1.+.012*math.sin(9*math.pi*t))
        target=Vector((.012*math.sin(5*math.pi*t),.3,1.015))
        camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.keyframe_insert(data_path='location',frame=frame);camera.keyframe_insert(data_path='rotation_euler',frame=frame)
    for i,obj in enumerate([o for o in scene.objects if o.name.startswith('background_highlight_')]):
        obj.location=((-1.1,-.25,.4,1.2)[i],3.25,(1.8,1.45,2.05,1.4)[i]-.5)
    # Two thin nonperiodic foreground stems provide a second spatial scale.
    for i,(x,z,angle) in enumerate([(-.10,1.10,.2),(.12,1.27,-.35)]):
        bpy.ops.mesh.primitive_cylinder_add(vertices=12,radius=.004,depth=.62,location=(x,.05,z),rotation=(0,angle,0))
        obj=bpy.context.object;obj.name=f'static_foreground_twig_{i}';obj.data.materials.append(natural_material(f'natural_twig_{i}'))
    scene.frame_set(1)


def export_scene(scene,out,label):
    objects=sorted([o for o in scene.objects if o.type=='MESH'],key=lambda x:x.name)
    names=[o.name for o in objects];foreground=[n for n in names if n in FOREGROUND or n.startswith('static_foreground_twig_')]
    arrays={};offset=0;face_offset=0;vertex_parts=[];face_parts=[];obj_parts=[];poly_parts=[]
    for i,obj in enumerate(objects,1):
        obj.pass_index=i;obj.data.calc_loop_triangles()
        v=np.array([x.co[:] for x in obj.data.vertices],np.float64);f=np.array([t.vertices[:] for t in obj.data.loop_triangles],np.int32)
        vertex_parts.append(v);face_parts.append(f+offset);obj_parts.append(np.full(len(f),i,np.int16));poly_parts.append(np.array([t.polygon_index for t in obj.data.loop_triangles],np.int32))
        arrays[f'object_{i}_vertices']=v;arrays[f'object_{i}_faces']=f;offset+=len(v);face_offset+=len(f)
    cameras=[];transforms=[]
    for frame in range(1,61):
        scene.frame_set(frame);cameras.append(np.array(scene.camera.matrix_world));transforms.append(np.stack([np.array(o.matrix_world) for o in objects]))
    fx=512*scene.camera.data.lens/scene.camera.data.sensor_width
    np.savez_compressed(out/'scene-geometry.npz',**arrays,object_names=np.array(names),object_ids=np.arange(1,len(names)+1),
        triangle_object_id=np.concatenate(obj_parts),triangle_polygon_id=np.concatenate(poly_parts),
        object_world_matrices=np.stack(transforms),camera_c2w_blender=np.stack(cameras),
        camera_c2w_opencv=np.stack(cameras)@np.diag([1,-1,-1,1]),intrinsics=np.array([[fx,0,256],[0,fx,256],[0,0,1.]]),
        foreground_object_ids=[names.index(n)+1 for n in foreground],dynamic_object_ids=[names.index('rigid_moving_sphere')+1],input_frames=INPUTS)
    scene.frame_set(1);scene.render.threads_mode='FIXED';scene.render.threads=12
    bpy.ops.wm.save_as_mainfile(filepath=str(out/f'{label}.blend'))
    metadata=dict(label=label,blender=bpy.app.version_string,resolution=[512,512],frames=60,input_frames=INPUTS,
        focus_distances_m=[5.3,8.4] if label=='B2-low' else [1.4,5.0],focal_length_mm=50,sensor_width_mm=36,
        foreground_names=foreground,dynamic_names=['rigid_moving_sphere'],objects=[dict(id=i+1,name=n) for i,n in enumerate(names)],
        asset_source='B1 original procedural primitives; high adds procedural noise/mineral/woodlike color and fine nonperiodic noise, no third-party assets',assets_license='CC0-1.0 generated scene assets',
        surface_id='stable object ID + triangulated face ID + barycentric/local coordinates; rigid object transforms per frame',
        rendering=dict(engine='Cycles',denoising=False,motion_blur=False,adaptive_sampling=False,linear_rgb='32-bit EXR',seed=20261110))
    (out/'scene.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')


def render(scene,out,args):
    if not args.pinhole and len(args.frames)>5:raise ValueError('B2 DOF is restricted to <=5 representative frames')
    scene.render.resolution_x=scene.render.resolution_y=args.size;scene.render.resolution_percentage=100
    scene.render.threads_mode='FIXED';scene.render.threads=12;scene.cycles.device=args.device
    scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False;scene.render.use_motion_blur=False
    if args.device=='GPU':
        pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
        for d in pref.devices:d.use=d.type=='OPTIX'
        if not any(d.use and d.type=='OPTIX' for d in pref.devices):raise RuntimeError('No OptiX GPU')
    original_visibility={o.name:o.visible_camera for o in scene.objects}
    metadata=json.loads((args.blend.parent/'scene.json').read_text())
    if args.hidden:
        for name in metadata['foreground_names']:bpy.data.objects[name].visible_camera=False
    rows=[];summary=out/'timings.json'
    if summary.exists():rows=json.loads(summary.read_text())
    for frame in args.frames:
        scene.frame_set(frame)
        for stop in args.fstops:
            for focus in args.focuses:
                for samples in args.samples:
                    lens='pinhole' if args.pinhole else f'N{stop:g}-F{focus:g}'
                    key=f'f{frame:03d}-{lens}-s{samples}-{args.size}-{args.device}-seed{args.seed}'+('-hidden' if args.hidden else '')
                    if any(r['key']==key for r in rows):continue
                    scene.camera.data.dof.use_dof=not args.pinhole;scene.camera.data.dof.aperture_fstop=stop;scene.camera.data.dof.focus_distance=focus
                    scene.cycles.samples=samples;scene.cycles.seed=args.seed
                    scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32';scene.render.image_settings.color_mode='RGB';scene.render.filepath=str(out/(key+'.exr'))
                    start=time.monotonic();bpy.ops.render.render(write_still=True);seconds=time.monotonic()-start
                    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8';bpy.data.images['Render Result'].save_render(str(out/(key+'.png')),scene=scene)
                    row=dict(key=key,frame=frame,fstop=stop,focus_m=focus,samples=samples,seed=args.seed,hidden=args.hidden,pinhole=args.pinhole,resolution=[args.size,args.size],device=args.device,elapsed_seconds=seconds,exr_bytes=(out/(key+'.exr')).stat().st_size)
                    rows.append(row);summary.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8');print('B2_RENDER',json.dumps(row),flush=True)
    for name,value in original_visibility.items():bpy.data.objects[name].visible_camera=value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['build','render'],required=True)
    p.add_argument('--source-blend',type=Path);p.add_argument('--blend',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label',choices=['B2-low','B2-high']);p.add_argument('--frames',type=int,nargs='+',default=[1,30])
    p.add_argument('--samples',type=int,nargs='+',default=[16]);p.add_argument('--fstops',type=float,nargs='+',default=[1.4]);p.add_argument('--focuses',type=float,nargs='+',default=[1.4])
    p.add_argument('--size',type=int,default=512);p.add_argument('--device',choices=['CPU','GPU'],default='CPU');p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--pinhole',action='store_true');p.add_argument('--hidden',action='store_true');a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='build':
        bpy.ops.wm.open_mainfile(filepath=str(a.source_blend.resolve()));scene=bpy.context.scene
        if a.label=='B2-high':high_scene(scene)
        export_scene(scene,a.output,a.label)
    else:
        a.blend=a.blend.resolve();bpy.ops.wm.open_mainfile(filepath=str(a.blend));render(bpy.context.scene,a.output,a)


if __name__=='__main__':main()
