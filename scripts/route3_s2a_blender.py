"""Second-batch static scenes. Pilot generator and outputs remain unchanged.

Original procedural assets only. Build in Blender 4.5.3; render with the
previous independently checked explicit-job renderer.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route3_blender_b1 import cube,material
from route3_pilot_blender import INPUTS,render,write


from bpy.app.handlers import persistent

@persistent
def ensure_object_local_aov(*args):
    """Also give emissive-only materials a correct unfiltered local-coordinate pass.

    Installed after loading an immutable scene; RGB shading is unchanged.
    """
    for m in bpy.data.materials:
        if not m.use_nodes:continue
        n=m.node_tree.nodes;l=m.node_tree.links
        if any(x.bl_idname=='ShaderNodeOutputAOV' and x.aov_name=='ObjectLocal' for x in n):continue
        geo=n.new('ShaderNodeNewGeometry');t=n.new('ShaderNodeVectorTransform');t.vector_type='POINT';t.convert_from='WORLD';t.convert_to='OBJECT';l.new(geo.outputs['Position'],t.inputs[0]);a=n.new('ShaderNodeOutputAOV');a.aov_name='ObjectLocal';l.new(t.outputs[0],a.inputs['Color'])

bpy.app.handlers.load_post.append(ensure_object_local_aov)


def textured(name,rng,kind,frequency):
    m=material(name,rng.uniform(.02,.8,3),.7);n=m.node_tree.nodes;l=m.node_tree.links;bs=n.get('Principled BSDF')
    coord=n.new('ShaderNodeTexCoord');mapping=n.new('ShaderNodeVectorMath');mapping.operation='MULTIPLY';mapping.inputs[1].default_value=(frequency,frequency,frequency);l.new(coord.outputs['Generated'],mapping.inputs[0])
    color0=np.array([.005,.005,.005]);color1=rng.uniform(.65,1.,3)
    if kind=='low':
        tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=3;color0=color1*.45
    elif kind=='noise':
        tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=1;tex.inputs['Detail'].default_value=0;tex.inputs['Roughness'].default_value=0
    elif kind=='stripes':
        tex=n.new('ShaderNodeTexWave');tex.wave_type='BANDS';tex.bands_direction=rng.choice(['X','Z','DIAGONAL']).item();tex.inputs['Scale'].default_value=1;tex.inputs['Distortion'].default_value=float(rng.uniform(.1,1.))
    else:
        tex=n.new('ShaderNodeTexChecker');tex.inputs['Scale'].default_value=1
        tex.inputs['Color1'].default_value=(*color0,1);tex.inputs['Color2'].default_value=(*color1,1)
    l.new(mapping.outputs[0],tex.inputs['Vector'])
    if kind in ['checker','blocks']:
        l.new(tex.outputs['Color'],bs.inputs['Base Color'])
    else:
        ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.interpolation='CONSTANT' if kind!='low' else 'LINEAR';ramp.color_ramp.elements[0].position=.43 if kind!='low' else 0;ramp.color_ramp.elements[1].position=.57 if kind!='low' else 1
        ramp.color_ramp.elements[0].color=(*color0,1);ramp.color_ramp.elements[1].color=(*color1,1);l.new(tex.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs['Color'],bs.inputs['Base Color'])
    geo=n.new('ShaderNodeNewGeometry');transform=n.new('ShaderNodeVectorTransform');transform.vector_type='POINT';transform.convert_from='WORLD';transform.convert_to='OBJECT';l.new(geo.outputs['Position'],transform.inputs[0]);aov=n.new('ShaderNodeOutputAOV');aov.aov_name='ObjectLocal';l.new(transform.outputs[0],aov.inputs['Color'])
    return m


def build(out,seed,index):
    if (out/'scene.blend').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene
    s.frame_start=1;s.frame_end=60;s.render.engine='CYCLES';s.render.resolution_x=s.render.resolution_y=512;s.render.resolution_percentage=100;s.render.fps=24;s.render.threads_mode='FIXED';s.render.threads=12
    s.cycles.use_denoising=False;s.cycles.use_adaptive_sampling=False;s.cycles.max_bounces=6;s.render.use_motion_blur=False;s.cycles.pixel_filter_type='BOX';s.cycles.filter_width=1.
    s.view_settings.view_transform='Standard';s.view_settings.look='None';s.view_settings.exposure=0;s.view_settings.gamma=1;s.world=bpy.data.worlds.new('World');s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.4,.43,.48,1);s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=float(rng.uniform(.35,.65))
    family=['thin','broad','multilayer'][index%3];low=index%5==4;styles=['low'] if low else ['checker','stripes','blocks','noise'];fgdepth=float(rng.uniform(1.28,1.48));bgdepth=float(rng.uniform(3.6,7.2));baseline=float(rng.uniform(.15,.45));yaw=float(rng.uniform(-.15,.15));bgwidth=bgdepth*.95
    kinds=[str(rng.choice(styles)) for _ in range(16)];freqs=rng.uniform(14,70,16);mats=[textured(f'seed{seed}-material{i}-{kinds[i]}',rng,kinds[i],float(freqs[i])) for i in range(16)]
    wall=cube('background_wall',(0,bgdepth-1.5,1.1),(bgwidth,.08,bgwidth),mats[0]);wall.rotation_euler[2]=yaw
    cube('background_floor',(0,bgdepth*.3,-.05),(bgwidth,bgdepth*2,.1),mats[1])
    # Different independent combinations and positions, never the B2 layout.
    if family=='thin':
        for i,x in enumerate(np.linspace(-.46,.46,8)+rng.uniform(-.025,.025,8)):
            cube(f'foreground_0_bar_{i}',(float(x),fgdepth-1.5+float(rng.uniform(-.05,.05)),1.1+float(rng.uniform(-.1,.1))),(float(rng.uniform(.018,.045)),.045,float(rng.uniform(.65,1.1))),mats[(i+2)%16])
        cube('foreground_0_diagonal',(0,fgdepth-1.45,.97),(.95,.05,.035),mats[10]).rotation_euler[1]=float(rng.uniform(-.3,.3))
    else:
        count=3 if family=='broad' else 4
        for i,x in enumerate(np.linspace(-.42,.42,count)+rng.uniform(-.035,.035,count)):
            d=fgdepth+(float(rng.uniform(.1,.45)) if family=='multilayer' and i%2 else 0)
            o=cube(f'foreground_{i%2}_panel_{i}',(float(x),d-1.5,1.1+float(rng.uniform(-.12,.12))),(float(rng.uniform(.14,.24)),.055,float(rng.uniform(.65,1.)) ),mats[i+2]);o.rotation_euler[1]=float(rng.uniform(-.2,.2))
        if family=='multilayer':cube('foreground_1_cross',(float(rng.uniform(-.15,.15)),fgdepth-1.1,1.25),(.65,.06,.09),mats[9])
    # Additional independently shaped rear assets and text-like bright blocks.
    for i in range(3):
        d=float(rng.uniform(max(fgdepth+.7,2.5),bgdepth-.15));cube(f'background_asset_{i}',(float(rng.uniform(-.8,.8)),d-1.5,float(rng.uniform(.7,1.5))),(float(rng.uniform(.13,.4)),float(rng.uniform(.07,.2)),float(rng.uniform(.2,.6))),mats[10+i])
    if not low:
        for i in range(9):
            cube(f'background_glyph_{i}',(float(rng.uniform(-bgwidth*.28,bgwidth*.28)),bgdepth-1.56,float(rng.uniform(.3,1.9))),(float(rng.uniform(.03,.13)),.015,float(rng.uniform(.03,.16))),mats[13+i%3])
        em=material('small_hidden_emitter',(.9,.7,.35),.3);bs=em.node_tree.nodes.get('Principled BSDF');bs.inputs['Emission Color'].default_value=(1,.6,.2,1);bs.inputs['Emission Strength'].default_value=3
        for i in range(3):cube(f'background_emitter_{i}',(float(rng.uniform(-.3,.3)),bgdepth-1.57,float(rng.uniform(.85,1.4))),(.012,.008,.018),em)
    bpy.ops.object.light_add(type='AREA',location=(-2,-2,4));bpy.context.object.data.energy=float(rng.uniform(300,600));bpy.context.object.data.size=4
    bpy.ops.object.camera_add();cam=bpy.context.object;s.camera=cam;cam.data.lens=50;cam.data.sensor_width=36;cam.data.sensor_fit='HORIZONTAL';cam.data.clip_end=100;cam.data.dof.aperture_blades=0;phase=float(rng.uniform(0,2*math.pi));forward=float(rng.uniform(.01,.05))
    for frame in range(1,61):
        t=(frame-1)/59;cam.location=(-baseline/2+baseline*t+.005*math.sin(7*math.pi*t+phase),-1.5+forward*math.sin(2*math.pi*t+phase),1.1+.009*math.sin(5*math.pi*t+phase));target=Vector((.01*math.sin(3*math.pi*t),3,1.1));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cam.keyframe_insert(data_path='location',frame=frame);cam.keyframe_insert(data_path='rotation_euler',frame=frame)
    objects=sorted([o for o in s.objects if o.type=='MESH'],key=lambda o:o.name);arrays={};parts=[];polys=[]
    for i,o in enumerate(objects,1):
        o.pass_index=i;o.data.calc_loop_triangles();arrays[f'object_{i}_vertices']=np.array([v.co[:] for v in o.data.vertices]);arrays[f'object_{i}_faces']=np.array([a.vertices[:] for a in o.data.loop_triangles],np.int32);parts.append(np.full(len(o.data.loop_triangles),i,np.int16));polys.append(np.array([a.polygon_index for a in o.data.loop_triangles],np.int32))
    cameras=[];transforms=[]
    for frame in range(1,61):s.frame_set(frame);cameras.append(np.array(cam.matrix_world));transforms.append(np.stack([np.array(o.matrix_world) for o in objects]))
    ids=np.arange(1,len(objects)+1);fg=[int(i) for i,o in zip(ids,objects) if o.name.startswith('foreground_')];np.savez_compressed(out/'scene-geometry.npz',**arrays,object_names=[o.name for o in objects],object_ids=ids,triangle_object_id=np.concatenate(parts),triangle_polygon_id=np.concatenate(polys),object_world_matrices=np.stack(transforms),camera_c2w_blender=np.stack(cameras),camera_c2w_opencv=np.stack(cameras)@np.diag([1,-1,-1,1]),intrinsics=[[512*50/36,0,256],[0,512*50/36,256],[0,0,1]],foreground_object_ids=fg,dynamic_object_ids=np.array([],np.int16),input_frames=INPUTS)
    layer=s.view_layers[0];layer.use_pass_z=True;layer.use_pass_position=True;layer.use_pass_object_index=True;layer.use_pass_cryptomatte_object=True;layer.pass_cryptomatte_depth=6;bpy.ops.scene.view_layer_add_aov();layer.aovs[-1].name='ObjectLocal';s.frame_set(1);ensure_object_local_aov();bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    write(out/'scene.json',dict(label=out.name,seed=seed,candidate_index=index,generator_version="v2-focus-support",family=family,low_texture=low,texture_kinds=kinds,texture_frequency=freqs.tolist(),foreground_depth_design_m=fgdepth,background_depth_design_m=bgdepth,foreground_background_depth_ratio=fgdepth/bgdepth,camera_translation_baseline_design_m=baseline,forward_amplitude_m=forward,blender=bpy.app.version_string,frames=60,input_frames=INPUTS,focus_distances_m=[1.4,5.],focal_length_mm=50,sensor_width_mm=36,objects=[dict(id=int(i),name=o.name) for i,o in zip(ids,objects)],foreground_names=[o.name for o in objects if o.name.startswith('foreground_')],dynamic_names=[],assets_source='independent original procedural primitives and high-contrast node patterns;not B2 layout',assets_license='CC0-1.0 generated assets',render=dict(filter_type='BOX',filter_width=1.,denoising=False,motion_blur=False,adaptive=False),static_object_transforms_verified=bool(np.allclose(transforms,transforms[0]))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path);p.add_argument('--seed',type=int);p.add_argument('--index',type=int);p.add_argument('--jobs',type=Path);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.build:build(a.build,a.seed,a.index)
    else:render(a.jobs)

if __name__=='__main__':main()
