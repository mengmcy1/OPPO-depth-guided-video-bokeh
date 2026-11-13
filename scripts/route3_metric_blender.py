"""Appendix8 metric Object textures, independent new procedural scenes.

The former generator is read-only. It supplies mesh/layout randomization;
this entry replaces only its material builder and saves a separate scene.
"""
import argparse,json,sys,shutil,math
from pathlib import Path
import bpy,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import route3_s2a_blender as previous
from route3_blender_b1 import material
from route3_pilot_blender import render,write

def metric_material(name,rng,kind,unused_frequency):
    m=material(name,rng.uniform(.08,.65,3),.7);nodes=m.node_tree.nodes;links=m.node_tree.links;bs=nodes.get('Principled BSDF')
    ref=bpy.data.objects.get('MetricTextureWorldReference')
    if ref is None:ref=bpy.data.objects.new('MetricTextureWorldReference',None);bpy.context.scene.collection.objects.link(ref)
    coord=nodes.new('ShaderNodeTexCoord');coord.object=ref;frequency=nodes.new('ShaderNodeVectorMath');frequency.name='MetricFrequency';frequency.operation='MULTIPLY';frequency.inputs[1].default_value=(1.,1.,1.);links.new(coord.outputs['Object'],frequency.inputs[0]);rotation=nodes.new('ShaderNodeVectorRotate');rotation.rotation_type='AXIS_ANGLE';rotation.inputs['Axis'].default_value=(0,1,0);rotation.inputs['Angle'].default_value=float(rng.uniform(0,2*math.pi));m['texture_rotation_radians']=rotation.inputs['Angle'].default_value;links.new(frequency.outputs[0],rotation.inputs['Vector'])
    base=rng.uniform(.12,.65,3);contrast=float(rng.uniform(.15,.30) if kind=='low' else rng.uniform(.3,.7));lo=np.clip(base*(1-contrast),.025,.95);hi=np.clip(base+(1-base)*contrast,.025,.95)
    if kind in ['noise','low']:
        tex=nodes.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=1;tex.inputs['Detail'].default_value=2;tex.inputs['Roughness'].default_value=.5;tex.inputs['Lacunarity'].default_value=2;links.new(rotation.outputs['Vector'],tex.inputs['Vector']);factor=tex.outputs['Fac'];m['metric_frequency_divisor']=4.;m['metric_texture_kind']='soft1/f_octaves'
    else:
        def wave(direction):
            tex=nodes.new('ShaderNodeTexWave');tex.wave_type='BANDS';tex.bands_direction=direction;tex.wave_profile='SIN';tex.inputs['Scale'].default_value=2*math.pi/20;tex.inputs['Distortion'].default_value=0;links.new(rotation.outputs['Vector'],tex.inputs['Vector']);return tex.outputs['Fac']
        factor=wave('X')
        if kind in ['checker','blocks']:
            a=nodes.new('ShaderNodeMath');a.operation='MULTIPLY_ADD';a.inputs[1].default_value=2;a.inputs[2].default_value=-1;links.new(factor,a.inputs[0]);b=nodes.new('ShaderNodeMath');b.operation='MULTIPLY_ADD';b.inputs[1].default_value=2;b.inputs[2].default_value=-1;links.new(wave('Z'),b.inputs[0]);c=nodes.new('ShaderNodeMath');c.operation='MULTIPLY';links.new(a.outputs[0],c.inputs[0]);links.new(b.outputs[0],c.inputs[1]);d=nodes.new('ShaderNodeMath');d.operation='MULTIPLY_ADD';d.inputs[1].default_value=.5;d.inputs[2].default_value=.5;links.new(c.outputs[0],d.inputs[0]);factor=d.outputs[0];m['metric_frequency_divisor']=math.sqrt(2);m['metric_texture_kind']='soft_checker'
        else:m['metric_frequency_divisor']=1.;m['metric_texture_kind']='sine_stripes'
    ramp=nodes.new('ShaderNodeValToRGB');ramp.color_ramp.interpolation='EASE';ramp.color_ramp.elements[0].color=(*lo,1);ramp.color_ramp.elements[1].color=(*hi,1);links.new(factor,ramp.inputs[0]);links.new(ramp.outputs[0],bs.inputs['Base Color']);m['metric_contrast']=contrast
    return m

def build(out,seed,index):
    if (out/'scene.blend').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);draft=out/'authoring'
    if draft.exists():
        trial=1
        while (out/f'authoring-retry-{trial:02d}').exists():trial+=1
        draft=out/f'authoring-retry-{trial:02d}'
    previous.textured=metric_material;previous.build(draft,seed,index);s=bpy.context.scene;g=np.load(draft/'scene-geometry.npz');cameras=g['camera_c2w_opencv'];objects=[];textures=[]
    for obj in [o for o in s.objects if o.type=='MESH']:
        corners=np.array(obj.bound_box)@np.array(obj.matrix_world)[:3,:3].T+np.array(obj.matrix_world)[:3,3];maxdepth=max(float(((corners-c[:3,3])@c[:3,:3])[:,2].max()) for c in cameras);cap=.85*(512*50/36)/(36*max(maxdepth,.2));cap=min(cap,.08) if obj.name=='background_floor' else cap
        for i,mat in enumerate(obj.data.materials):
            if not mat or 'metric_texture_kind' not in mat:continue
            own=mat.copy();own.name=mat.name+'-'+obj.name;obj.data.materials[i]=own;f=cap/float(own['metric_frequency_divisor']);own.node_tree.nodes['MetricFrequency'].inputs[1].default_value=(f,f,f);textures.append(dict(object=obj.name,kind=own['metric_texture_kind'],maximum_feature_cycles_per_m=cap,coordinate_cycles_per_m=f,contrast=float(own['metric_contrast']),maximum_camera_depth_m=maxdepth,coordinate='Object with identity world reference;meters independent of mesh scale',minimum_nongrazing_design_period_px=36*math.cos(math.radians(75))/.85))
        objects.append(obj)
    shutil.copyfile(draft/'scene-geometry.npz',out/'scene-geometry.npz');meta=json.loads((draft/'scene.json').read_text());meta.update(label=out.name,generator_version='metricObject-thirdtexture-v1',texture_version='附八第三版纹理:metric Object soft textures',textures=textures,has_highlights=any('emitter' in o.name or o.name.startswith('background_highlight') for o in objects),legacy_texture_frequency_draws_not_active=True,old_generator_behavior_unchanged=True,authoring_draft_retained=str(draft),texture_period_verified_by_world_pass_before_methods=False,filter_type='BOX',filter_width=1.,assets_source='independent procedural shapes and metric Object soft textures;no external assets');write(out/'scene.json',meta);bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'));print('METRIC_BUILD',out.name,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path);p.add_argument('--seed',type=int);p.add_argument('--index',type=int);p.add_argument('--jobs',type=Path);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);build(a.build,a.seed,a.index) if a.build else render(a.jobs)

if __name__=='__main__':main()
