"""Appendix-seven static pilots: procedural scenes and explicit sparse render jobs.

Run in the pinned Blender binary. Never writes an existing job output.
Color uses a recorded box filter; Position/Z/ObjectIndex are data passes.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Vector, Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route3_blender_b1 import cube, material

INPUTS=[1,6,11,16,21,26,30,35,40,45,50,55,60]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def textured(name,rng):
    color=rng.uniform(.08,.65,3);m=material(name,color,.65)
    n=m.node_tree.nodes;l=m.node_tree.links;bs=n.get('Principled BSDF')
    coord=n.new('ShaderNodeTexCoord');tex=n.new('ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value=float(rng.uniform(12,42));tex.inputs['Detail'].default_value=3
    l.new(coord.outputs['Object'],tex.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(*np.maximum(.015,color*.3),1)
    ramp.color_ramp.elements[1].color=(*color,1);l.new(tex.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs[0],bs.inputs['Base Color'])
    geo=n.new('ShaderNodeNewGeometry');transform=n.new('ShaderNodeVectorTransform')
    transform.vector_type='POINT';transform.convert_from='WORLD';transform.convert_to='OBJECT'
    l.new(geo.outputs['Position'],transform.inputs[0]);aov=n.new('ShaderNodeOutputAOV');aov.name='ObjectLocal';aov.aov_name='ObjectLocal';l.new(transform.outputs[0],aov.inputs['Color'])
    return m


def build(out,label,seed):
    if (out/'scene.blend').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);bpy.ops.wm.read_factory_settings(use_empty=True)
    s=bpy.context.scene;s.frame_start=1;s.frame_end=60;s.render.engine='CYCLES'
    s.render.resolution_x=s.render.resolution_y=512;s.render.resolution_percentage=100
    s.render.fps=24;s.render.threads_mode='FIXED';s.render.threads=12
    s.cycles.use_denoising=False;s.cycles.use_adaptive_sampling=False;s.cycles.max_bounces=6
    s.render.use_motion_blur=False;s.cycles.pixel_filter_type='BOX';s.cycles.filter_width=1.
    s.view_settings.view_transform='Standard';s.view_settings.look='None';s.view_settings.exposure=0;s.view_settings.gamma=1
    s.world=bpy.data.worlds.new('World');s.world.use_nodes=True
    s.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.20,.23,.28,1)
    s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=float(rng.uniform(.35,.55))
    mats=[textured(f'{label}-original-material-{i}',rng) for i in range(12)]
    cube('background_wall',(0,3.45,1.15),(6.,.08,3.0),mats[0])
    cube('background_floor',(0,1.,-.12),(6.,8.,.1),mats[1])
    # Distinct asset combinations, not alternate views of the same scene.
    if label=='pilot-thin':
        for i,x in enumerate([-.43,-.16,.16,.43]):
            bpy.ops.mesh.primitive_cylinder_add(vertices=16,radius=float(rng.uniform(.009,.02)),depth=float(rng.uniform(.7,1.1)),location=(x,float(rng.uniform(-.08,.12)),1.))
            o=bpy.context.object;o.name=f'foreground_0_stem_{i}';o.rotation_euler[1]=float(rng.uniform(-.25,.25));o.data.materials.append(mats[i+2])
        cube('foreground_0_crossbar',(0,-.05,.75),(.95,.06,.055),mats[7])
        cube('foreground_1_backpost',(.13,.62,1.12),(.095,.08,.8),mats[8])
        for i in range(3):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=.18,location=(-.64+.61*i,2.+.16*i,.72+.28*i))
            o=bpy.context.object;o.name=f'background_static_sphere_{i}';o.data.materials.append(mats[9+i])
    else:
        # Three boards and gaps form a large mask with a second depth layer.
        for i,(x,z,w,h) in enumerate([(-.40,1.03,.27,.95),(.38,1.12,.25,.8),(0,.77,.56,.12)]):
            cube(f'foreground_0_panel_{i}',(x,-.10,z),(w,.07,h),mats[i+2])
        cube('foreground_1_rear_panel',(.05,.50,1.25),(.32,.08,.42),mats[6])
        for i in range(3):
            cube(f'background_shelf_{i}',(-.65+.65*i,2.3,1.2+.18*i),(.3,.2,.55),mats[8+i])
    bpy.ops.object.light_add(type='AREA',location=(-2,-2,4.));light=bpy.context.object
    light.data.energy=float(rng.uniform(350,550));light.data.size=4
    bpy.ops.object.camera_add();cam=bpy.context.object;s.camera=cam
    cam.data.lens=50;cam.data.sensor_width=36;cam.data.sensor_fit='HORIZONTAL'
    cam.data.dof.aperture_blades=0;cam.data.clip_end=100
    phase=float(rng.uniform(0,2*math.pi))
    for frame in range(1,61):
        t=(frame-1)/59;cam.location=(-.14+.28*t+.008*math.sin(9*math.pi*t+phase),-1.5+.016*math.sin(3*math.pi*t),1.+.012*math.sin(7*math.pi*t+phase))
        target=Vector((.012*math.sin(4*math.pi*t),.35,1.02))
        cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
        cam.keyframe_insert(data_path='location',frame=frame);cam.keyframe_insert(data_path='rotation_euler',frame=frame)
    objects=sorted([o for o in s.objects if o.type=='MESH'],key=lambda o:o.name);names=[o.name for o in objects]
    arrays={};parts=[];polys=[]
    for i,o in enumerate(objects,1):
        o.pass_index=i;o.data.calc_loop_triangles();v=np.array([a.co[:] for a in o.data.vertices]);f=np.array([a.vertices[:] for a in o.data.loop_triangles],np.int32)
        arrays[f'object_{i}_vertices']=v;arrays[f'object_{i}_faces']=f;parts.append(np.full(len(f),i,np.int16));polys.append(np.array([a.polygon_index for a in o.data.loop_triangles],np.int32))
    cameras=[];transforms=[]
    for frame in range(1,61):
        s.frame_set(frame);cameras.append(np.array(cam.matrix_world));transforms.append(np.stack([np.array(o.matrix_world) for o in objects]))
    ids=np.arange(1,len(objects)+1);fg=[i for i,o in zip(ids,objects) if o.name.startswith('foreground_')]
    if not (out/'scene-geometry.npz').exists():
        np.savez_compressed(out/'scene-geometry.npz',**arrays,object_names=names,object_ids=ids,triangle_object_id=np.concatenate(parts),triangle_polygon_id=np.concatenate(polys),object_world_matrices=np.stack(transforms),camera_c2w_blender=np.stack(cameras),camera_c2w_opencv=np.stack(cameras)@np.diag([1,-1,-1,1]),intrinsics=[[512*50/36,0,256],[0,512*50/36,256],[0,0,1]],foreground_object_ids=fg,dynamic_object_ids=np.array([],np.int16),input_frames=INPUTS)
    layer=s.view_layers[0];layer.use_pass_z=True;layer.use_pass_position=True;layer.use_pass_object_index=True
    layer.use_pass_cryptomatte_object=True;layer.pass_cryptomatte_depth=6
    bpy.ops.scene.view_layer_add_aov();layer.aovs[-1].name='ObjectLocal';s.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    write(out/'scene.json',dict(label=label,seed=seed,blender=bpy.app.version_string,frames=60,input_frames=INPUTS,focus_distances_m=[1.4,5.],focal_length_mm=50,sensor_width_mm=36,objects=[dict(id=i,name=n) for i,n in zip(ids.tolist(),names)],foreground_names=[o.name for o in objects if o.name.startswith('foreground_')],dynamic_names=[],assets_source='independent original procedural primitives and node textures;no downloaded assets',assets_license='CC0-1.0 generated assets',render=dict(filter_type='BOX',filter_width=1.,denoising=False,motion_blur=False,adaptive=False),static_object_transforms_verified=bool(np.allclose(transforms,transforms[0]))))


def file_output(scene,socket,path,color_mode='RGB'):
    if socket.type=='VECTOR':
        separate=scene.node_tree.nodes.new('CompositorNodeSeparateXYZ');combine=scene.node_tree.nodes.new('CompositorNodeCombineColor');combine.mode='RGB'
        scene.node_tree.links.new(socket,separate.inputs[0])
        for i in range(3):scene.node_tree.links.new(separate.outputs[i],combine.inputs[i])
        socket=combine.outputs[0]
    n=scene.node_tree.nodes.new('CompositorNodeOutputFile');n.base_path=str(path.parent)
    n.format.file_format='OPEN_EXR';n.format.color_mode=color_mode;n.format.color_depth='32';n.format.exr_codec='ZIP'
    n.file_slots[0].path=path.name+'-';scene.node_tree.links.new(socket,n.inputs[0]);return str(path)+'-'+f'{scene.frame_current:04d}'+'.exr'


def render(jobs_path):
    jobs=json.loads(jobs_path.read_text());rows=[];manifest=jobs_path.with_name(jobs_path.stem+'-results.json')
    if manifest.exists():rows=json.loads(manifest.read_text())
    done={r['id'] for r in rows};current=None;original={};original_camera=None;lens_camera=None
    for j in jobs:
        if j['id'] in done:continue
        out=Path(j['output']);out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():raise FileExistsError(out)
        if current!=j['blend']:
            bpy.ops.wm.open_mainfile(filepath=j['blend']);current=j['blend'];scene=bpy.context.scene
            original={o.name:o.visible_camera for o in scene.objects};original_camera=scene.camera
            lens_camera=original_camera.copy();lens_camera.data=original_camera.data.copy();lens_camera.animation_data_clear();lens_camera.data.animation_data_clear();lens_camera.parent=None;scene.collection.objects.link(lens_camera)
        s=bpy.context.scene;s.camera=original_camera;s.frame_set(j['frame'])
        for o in s.objects:o.visible_camera=original.get(o.name,True)
        for name in j.get('hide_names',[]):bpy.data.objects[name].visible_camera=False
        s.render.engine='CYCLES';s.cycles.device='GPU';s.render.threads_mode='FIXED';s.render.threads=12
        pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
        for d in pref.devices:d.use=d.type=='OPTIX'
        if not any(d.use for d in pref.devices):raise RuntimeError('No OptiX GPU')
        s.render.resolution_x=s.render.resolution_y=j.get('size',512);s.render.resolution_percentage=100
        s.cycles.samples=j['spp'];s.cycles.seed=j.get('seed',20261112);s.cycles.use_denoising=False;s.cycles.use_adaptive_sampling=False;s.render.use_motion_blur=False
        s.cycles.pixel_filter_type='BOX';s.cycles.filter_width=1.
        base=s.camera.matrix_world.copy();shift=(s.camera.data.shift_x,s.camera.data.shift_y)
        if 'aperture_xy' in j:
            s.camera=lens_camera
            # CV aperture Y is down, Blender camera Y is up.
            ax,ay=j['aperture_xy'];s.camera.matrix_world=base@Matrix.Translation((ax,-ay,0))
            s.camera.data.shift_x=-ax/(.036*j['focus']/.05)
            s.camera.data.shift_y=ay/(.036*j['focus']/.05)
        s.camera.data.dof.use_dof=j.get('dof',False);s.camera.data.dof.aperture_fstop=j.get('fstop',1.4);s.camera.data.dof.focus_distance=j.get('focus',1.4)
        s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_mode='RGBA';s.render.image_settings.color_depth='32';s.render.image_settings.exr_codec='ZIP';s.render.film_transparent=j.get('transparent_world',False)
        s.render.filepath=str(out);s.use_nodes=True;s.node_tree.nodes.clear();layer=s.node_tree.nodes.new('CompositorNodeRLayers');passes={}
        if j.get('data_passes',False):
            for name in ['Position','Depth','IndexOB','ObjectLocal']:
                passes[name]=file_output(s,layer.outputs[name],out.with_suffix('').with_name(out.stem+'-'+name))
        if j.get('cryptomatte',False):
            # Preserve raw Cryptomatte float bit patterns in a multilayer EXR.
            n=s.node_tree.nodes.new('CompositorNodeOutputFile');n.base_path=str(out.with_suffix(''))+'-crypto-';n.format.file_format='OPEN_EXR_MULTILAYER';n.format.color_depth='32';n.file_slots.clear()
            for name in ['CryptoObject00','CryptoObject01','CryptoObject02']:
                n.layer_slots.new(name);s.node_tree.links.new(layer.outputs[name],n.inputs[name])
            n.file_slots[0].path=out.stem+'-crypto-';passes['Cryptomatte']=str(out.parent/(out.stem+'-crypto-'+f'{j["frame"]:04d}'+'.exr'))
        comp=s.node_tree.nodes.new('CompositorNodeComposite');s.node_tree.links.new(layer.outputs['Image'],comp.inputs[0])
        start=time.monotonic();bpy.ops.render.render(write_still=True);elapsed=time.monotonic()-start
        s.camera.matrix_world=base;s.camera.data.shift_x,s.camera.data.shift_y=shift
        row=dict(**j,passes=passes,seconds=elapsed,bytes=out.stat().st_size+sum(Path(p).stat().st_size for p in passes.values() if Path(p).exists()),blender=bpy.app.version_string,filter_type='BOX',filter_width=1.)
        rows.append(row);write(manifest,rows);print('PILOT_RENDER',j['id'],round(elapsed,3),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path);p.add_argument('--label');p.add_argument('--seed',type=int);p.add_argument('--jobs',type=Path)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.build:build(a.build,a.label,a.seed)
    else:render(a.jobs)


if __name__=='__main__':main()
