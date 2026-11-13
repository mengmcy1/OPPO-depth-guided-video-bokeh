"""Read-only scene inspection and principal-point phase/noise probes."""
import argparse,json,sys,time
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route3_pilot_blender import file_output

def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')

def linear_coordinate(socket,obj):
    inv=np.linalg.inv(np.array(obj.matrix_world))[:3,:3]
    if not socket.is_linked:return inv,'unlinked-generated-assumption'
    link=socket.links[0];n=link.from_node
    if n.type=='TEX_COORD':
        if link.from_socket.name=='Generated':
            bb=np.array(obj.bound_box);return np.diag(1/np.maximum(np.ptp(bb,axis=0),1e-8))@inv,'Generated'
        if link.from_socket.name=='Object':
            reference=n.object
            return (np.linalg.inv(np.array(reference.matrix_world))[:3,:3] if reference is not None else inv),'Object:'+reference.name if reference is not None else 'Object'
        raise ValueError('Unsupported texture coordinate '+link.from_socket.name)
    if n.type=='VECT_MATH' and n.operation=='MULTIPLY':
        a,label=linear_coordinate(n.inputs[0],obj);return np.diag(n.inputs[1].default_value)@a,label
    if n.bl_idname=='ShaderNodeVectorRotate' and n.rotation_type=='AXIS_ANGLE':
        a,label=linear_coordinate(n.inputs['Vector'],obj);rot=np.array(Matrix.Rotation(n.inputs['Angle'].default_value,3,n.inputs['Axis'].default_value));return rot@a,label
    raise ValueError('Unsupported coordinate chain '+n.bl_idname)

def inspect_materials(scene,objects):
    rows=[]
    for spec in objects:
        obj=bpy.data.objects[str(spec['name'])];obj.pass_index=int(spec['id']);patterns=[]
        for mat in obj.data.materials:
            if not mat or not mat.use_nodes:continue
            for n in mat.node_tree.nodes:
                if n.type not in ['TEX_CHECKER','TEX_NOISE','TEX_WAVE']:continue
                try:A,coords=linear_coordinate(n.inputs['Vector'],obj)
                except ValueError as e:patterns.append(dict(material=mat.name,node=n.name,error=str(e)));continue
                scale=float(n.inputs['Scale'].default_value);detail=float(n.inputs['Detail'].default_value) if 'Detail' in n.inputs else 0
                lac=float(n.inputs['Lacunarity'].default_value) if 'Lacunarity' in n.inputs else 2.
                if n.type=='TEX_CHECKER':vectors=A*scale/2;definition='checker full2cell repeat;hard edges have unlimited harmonics'
                elif n.type=='TEX_NOISE':vectors=A*scale*lac**np.ceil(detail);definition='nonperiodic finest nonzero octave feature-scale proxy;not a literal spectral cutoff'
                else:
                    direction=n.bands_direction if n.wave_type=='BANDS' else n.rings_direction
                    weights=np.array([1.,0,0]) if direction=='X' else np.array([0.,1,0]) if direction=='Y' else np.array([0.,0,1]) if direction=='Z' else np.ones(3)
                    coefficient=10 if direction=='DIAGONAL' and n.wave_type=='BANDS' else 20
                    vectors=(weights@A)[None]*scale*coefficient/(2*np.pi);definition='wave principal-phase period;distortion separately recorded'
                patterns.append(dict(material=mat.name,node=n.name,type=n.type,coordinate=coords,world_phase_gradient_cycles_m=vectors.tolist(),scale=scale,detail=detail,lacunarity=lac,distortion=float(n.inputs['Distortion'].default_value) if 'Distortion' in n.inputs else 0,bands_direction=n.bands_direction if n.type=='TEX_WAVE' else None,definition=definition))
        rows.append(dict(**spec,dimensions_m=list(obj.dimensions),matrix_world=np.array(obj.matrix_world).tolist(),patterns=patterns))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--metadata-only',action='store_true');a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);plan=json.loads(a.plan.read_text());rows=[]
    for case in plan:
        folder=Path(case['output']);folder.mkdir(parents=True,exist_ok=True);manifest=folder/'manifest.json'
        if manifest.exists() and not a.metadata_only:rows.append(json.loads(manifest.read_text()));continue
        bpy.ops.wm.open_mainfile(filepath=case['blend']);s=bpy.context.scene;s.frame_set(30);original=s.camera;pose=original.matrix_world.copy();sx=original.data.shift_x;sy=original.data.shift_y
        camera=original.copy();camera.data=original.data.copy();camera.animation_data_clear();camera.data.animation_data_clear();camera.parent=None;s.collection.objects.link(camera);camera.matrix_world=pose;s.camera=camera
        objects=inspect_materials(s,case['objects'])
        if a.metadata_only:
            correction=dict(scene=case['scene'],objects=objects,source='https://raw.githubusercontent.com/blender/blender/v4.5.3/intern/cycles/kernel/svm/wave.h',correction='diagonal bands coefficient10;axis bands20;fractional noise includes ceil(detail) octave');write(folder/'materials-corrected.json',correction);rows.append(correction);continue
        s.render.engine='CYCLES';s.cycles.device='GPU';prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
        for d in prefs.devices:d.use=d.type=='OPTIX'
        if not any(d.use for d in prefs.devices):raise RuntimeError('No allocated OptiX0')
        s.render.resolution_x=s.render.resolution_y=512;s.render.resolution_percentage=100;s.render.threads_mode='FIXED';s.render.threads=12;s.cycles.samples=256;s.cycles.use_denoising=False;s.cycles.use_adaptive_sampling=False;s.cycles.pixel_filter_type='BOX';s.cycles.filter_width=1.;s.render.use_motion_blur=False;camera.data.dof.use_dof=False
        s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_mode='RGBA';s.render.image_settings.color_depth='32';s.render.image_settings.exr_codec='ZIP';s.render.film_transparent=False
        result=[]
        for label,delta,seed in [('center',0.,20261112),('shift',.5/512,20261112),('noise',0.,2026111202)]:
            path=folder/(label+'.exr')
            if path.exists():raise FileExistsError(path)
            camera.matrix_world=pose;camera.data.shift_x=sx+delta;camera.data.shift_y=sy;s.cycles.seed=seed;s.render.filepath=str(path);s.use_nodes=True;s.node_tree.nodes.clear();layer=s.node_tree.nodes.new('CompositorNodeRLayers');comp=s.node_tree.nodes.new('CompositorNodeComposite');s.node_tree.links.new(layer.outputs['Image'],comp.inputs[0]);passes={}
            if label=='center':
                s.view_layers[0].use_pass_position=True;s.view_layers[0].use_pass_object_index=True
                bpy.context.view_layer.update();s.node_tree.nodes.remove(layer);layer=s.node_tree.nodes.new('CompositorNodeRLayers');s.node_tree.links.new(layer.outputs['Image'],comp.inputs[0])
                for name in ['Position','IndexOB']:passes[name]=file_output(s,layer.outputs[name],folder/(label+'-'+name))
            start=time.monotonic();bpy.ops.render.render(write_still=True)
            result.append(dict(label=label,path=str(path),shift_x=float(camera.data.shift_x),shift_y=float(camera.data.shift_y),expected_image_delta_xy=[-delta*512,0.],matrix_world=np.array(camera.matrix_world).tolist(),max_extrinsic_change=float(np.abs(np.array(camera.matrix_world)-np.array(pose)).max()),seed=seed,seconds=time.monotonic()-start,passes=passes))
        row=dict(scene=case['scene'],frame=30,objects=objects,rows=result,blender=bpy.app.version_string,filter='BOX1.0',samples=256,denoising=False,motion_blur=False,material_and_geometry_changed=False);write(manifest,row);rows.append(row);print('ALIAS_PROBE',case['scene'],flush=True)
    write(a.plan.with_name('alias-material-corrections.json' if a.metadata_only else 'alias-render-results.json'),dict(rows=rows))

if __name__=='__main__':main()
