"""Run inside Blender 4.5.3: procedural B1 scene, sparse renders, surface GT.

No external assets; all meshes/materials/animations are constructed here.
Rigid animation preserves polygon IDs and object-local surface coordinates.
This entry point deliberately has no full-sequence rendering option.
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
from bpy_extras.object_utils import world_to_camera_view


def material(name, color, roughness=.45, metallic=0.):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    return m


def cube(name, location, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object; obj.name = name; obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat); return obj


def build_scene(output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene; scene.frame_start = 1; scene.frame_end = 60; scene.render.fps = 24
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.render.threads_mode = 'FIXED'; scene.render.threads = 16
    scene.cycles.use_denoising = False; scene.cycles.use_adaptive_sampling = False
    scene.render.use_motion_blur = False; scene.cycles.max_bounces = 8
    scene.render.resolution_x = scene.render.resolution_y = 512; scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'OPEN_EXR'; scene.render.image_settings.color_mode = 'RGB'; scene.render.image_settings.color_depth = '32'
    scene.view_settings.view_transform = 'Standard'; scene.view_settings.look = 'None'; scene.view_settings.exposure = 0; scene.view_settings.gamma = 1
    scene.world = bpy.data.worlds.new('World'); scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.14, .17, .22, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .4
    floor = material('floor', (.2, .23, .25)); occluder = material('static_occluder', (.05, .11, .18))
    board = material('procedural_checker', (.6, .5, .3)); nodes = board.node_tree.nodes; links = board.node_tree.links
    tex = nodes.new('ShaderNodeTexChecker'); tex.inputs['Color1'].default_value = (.65, .08, .03, 1)
    tex.inputs['Color2'].default_value = (.03, .45, .6, 1); tex.inputs['Scale'].default_value = 12
    coord = nodes.new('ShaderNodeTexCoord'); links.new(coord.outputs['Generated'], tex.inputs['Vector'])
    links.new(tex.outputs['Color'], nodes.get('Principled BSDF').inputs['Base Color'])
    cube('floor', (0, .8, -.06), (7, 9, .12), floor)
    cube('hidden_checker_background', (0, 2.8, 1.7), (5, .1, 3.4), board)
    cube('static_foreground_left', (-.72, -.4, 1.25), (.18, .18, 2.5), occluder)
    cube('static_foreground_right', (.88, -.1, 1.25), (.18, .18, 2.5), occluder)
    cube('static_foreground_crossbar', (.05, -.3, .7), (1.8, .15, .12), occluder)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=.36, location=(-.9, .6, .9))
    moving = bpy.context.object; moving.name = 'rigid_moving_sphere'
    moving.data.materials.append(material('moving_gold', (.8, .36, .04), .23, .55))
    for poly in moving.data.polygons: poly.use_smooth = True
    bright = material('bright_emission', (1, 1, 1)); bsdf = bright.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Emission Color'].default_value = (1, .65, .25, 1); bsdf.inputs['Emission Strength'].default_value = 8
    for i, (x, z) in enumerate([(-1.1, 1.8),(-.25,1.45),(.4,2.05),(1.2,1.4)]):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=.035, location=(x, 2.65, z))
        bpy.context.object.name = f'background_highlight_{i}'; bpy.context.object.data.materials.append(bright)
    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 5)); light = bpy.context.object
    light.data.energy = 600; light.data.shape = 'DISK'; light.data.size = 4
    bpy.ops.object.camera_add(location=(0, -5.6, 1.5)); camera = bpy.context.object; scene.camera = camera
    camera.data.lens = 50; camera.data.sensor_width = 36; camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.dof.use_dof = True; camera.data.dof.aperture_blades = 0
    for frame in range(1, 61):
        t = (frame - 1) / 59
        moving.location = (-1.05 + 2.1 * t, .6 + .13 * math.sin(2 * math.pi * t), .95 + .1 * math.sin(4 * math.pi * t))
        moving.rotation_euler = (.2 * t, .6 * t, 2 * t)
        moving.keyframe_insert(data_path='location', frame=frame); moving.keyframe_insert(data_path='rotation_euler', frame=frame)
        camera.location = (-.3 + .6 * t + .025 * math.sin(13 * math.pi * t), -5.6 + .08 * math.sin(3 * math.pi * t), 1.5 + .04 * math.sin(9 * math.pi * t))
        target = Vector((.04 * math.sin(5 * math.pi * t), .7, 1.1))
        camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera.keyframe_insert(data_path='location', frame=frame); camera.keyframe_insert(data_path='rotation_euler', frame=frame)
    for i, obj in enumerate(sorted((x for x in scene.objects if x.type == 'MESH'), key=lambda x:x.name), 1):obj.pass_index = i
    layer = scene.view_layers[0]; layer.use_pass_z = True; layer.use_pass_object_index = True
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'b1-procedural.blend'))
    metadata = dict(blender=bpy.app.version_string, frames=60, resolution=[512,512], camera_lens_mm=50,
        sensor_width_mm=36, apertures_fstops=[.5,1.4,2.8,5.6], focus_distances_m=[5.3,8.4],
        scene_assets='original procedural primitives, node materials and rigid animation generated by this script; no third-party assets',
        assets_license='CC0-1.0 dedication for generated scene assets; no downloaded asset license dependency',
        render=dict(engine='Cycles',denoising=False,motion_blur=False,adaptive_sampling=False,max_bounces=8,
                    linear_output='32-bit scene-linear RGB EXR',preview='Standard view transform PNG',cpu_threads=16),
        objects=[dict(name=o.name,id=o.pass_index,polygons=len(o.data.polygons)) for o in scene.objects if o.type=='MESH'])
    (output / 'scene.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return scene


def render_sparse(scene, output, frames, samples, fstops, focuses, size, device, seed=20261110, pinhole=False):
    if len(frames)>4:raise ValueError('B1 accepts at most four representative frames')
    if any(f<1 or f>60 for f in frames):raise ValueError('Representative frame must be in 1..60')
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.cycles.device = device
    if device == 'GPU':
        pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
        for d in pref.devices:d.use=d.type=='OPTIX'
        if not any(d.use and d.type=='OPTIX' for d in pref.devices):
            raise RuntimeError('No enabled OptiX GPU; refusing silent CPU fallback')
    rows=[]; summary=output/'timings.json'
    if summary.exists():rows=json.loads(summary.read_text())
    for frame in frames:
        scene.frame_set(frame)
        for fstop in fstops:
            for focus in focuses:
                scene.camera.data.dof.use_dof = not pinhole;scene.camera.data.dof.aperture_fstop=fstop;scene.camera.data.dof.focus_distance=focus
                for spp in samples:
                    lens='pinhole' if pinhole else f'N{fstop:g}-F{focus:g}'
                    key=f'f{frame:03d}-{lens}-s{spp}-{size}-{device}'
                    if seed!=20261110:key+=f'-seed{seed}'
                    if any(r['key']==key for r in rows):continue
                    scene.cycles.samples=spp;scene.cycles.seed=seed
                    scene.render.filepath=str(output/(key+'.exr'))
                    started=time.monotonic();bpy.ops.render.render(write_still=True);seconds=time.monotonic()-started
                    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8'
                    bpy.data.images['Render Result'].save_render(str(output/(key+'.png')), scene=scene)
                    scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32'
                    row=dict(key=key,frame=frame,fstop=fstop,focus_m=focus,samples=spp,resolution=[size,size],device=device,
                             seed=seed,pinhole=pinhole,device_backend='OptiX' if device=='GPU' else 'CPU',
                             elapsed_seconds_including_render_and_exr_write=seconds,exr_bytes=(output/(key+'.exr')).stat().st_size)
                    rows.append(row);summary.write_text(json.dumps(rows,indent=2)+'\n');print('B1_TIMING',json.dumps(row),flush=True)


def surface_gt(scene, output, frame=30, size=64, samples=16, fstop=1.4, focus=5.3):
    """First-hit visibility with stable (object, polygon, local position) IDs.

    This rigid-scene prototype excludes points outside the center image from
    center-occlusion GT; crossing-frame queries use actual surface positions.
    """
    started=time.monotonic();scene.frame_set(frame);deps=bpy.context.evaluated_depsgraph_get();cam=scene.camera
    matrix=cam.matrix_world.copy();rotation=matrix.to_3x3();origin=matrix.translation
    fx=size*cam.data.lens/cam.data.sensor_width;aperture=cam.data.lens/1000/(2*fstop)
    half=samples//2;idx=np.arange(half);r=np.sqrt((idx+.5)/half);theta=idx*math.pi*(3-math.sqrt(5))
    disk=np.stack((r*np.cos(theta),r*np.sin(theta)),-1);disk=np.concatenate((disk,-disk))*aperture
    objects={o.name:o for o in scene.objects if o.type=='MESH'};object_ids={n:o.pass_index for n,o in objects.items()}
    ids=np.zeros((samples,size,size),np.int16);faces=np.full_like(ids,-1);local=np.full((samples,size,size,3),np.nan,np.float32)
    known=np.zeros_like(ids,bool);hidden=np.zeros_like(ids,bool);recover=np.zeros_like(ids,bool);candidates=[]
    aperture_depth=np.zeros_like(ids,np.float32)
    center_depth=np.zeros((size,size),np.float32);center_ids=np.zeros((size,size),np.int16)
    center_faces=np.full((size,size),-1,np.int32);center_local=np.full((size,size,3),np.nan,np.float32)
    for y in range(size):
        for x in range(size):
            direction=rotation@Vector(((x+.5-size/2)/fx,-(y+.5-size/2)/fx,-1.))
            hit=scene.ray_cast(deps,origin,direction.normalized())
            if hit[0]:
                obj=objects[hit[4].original.name];point=hit[1]
                center_depth[y,x]=-(matrix.inverted()@point).z
                center_ids[y,x]=object_ids[obj.name];center_faces[y,x]=hit[3]
                center_local[y,x]=obj.matrix_world.inverted()@point
    for k,xy in enumerate(disk):
        offset=Vector((float(xy[0]),float(xy[1]),0));ap_origin=origin+rotation@offset
        for y in range(size):
            for x in range(size):
                focal=Vector(((x+.5-size/2)/fx*focus,-(y+.5-size/2)/fx*focus,-focus))
                direction=rotation@(focal-offset)
                ok,point,normal,face,obj,_=scene.ray_cast(deps,ap_origin,direction.normalized())
                if not ok:continue
                obj=objects[obj.original.name];name=obj.name;ids[k,y,x]=object_ids[name];faces[k,y,x]=face
                lp=obj.matrix_world.inverted()@point;local[k,y,x]=lp
                aperture_depth[k,y,x]=-(matrix.inverted()@point).z
                uv=world_to_camera_view(scene,cam,point)
                if not (0<=uv.x<=1 and 0<=uv.y<=1 and uv.z>0):continue
                central=scene.ray_cast(deps,origin,(point-origin).normalized());known[k,y,x]=central[0]
                if central[0] and (central[1]-point).length>1e-4:
                    hidden[k,y,x]=True;candidates.append((k,y,x,name,lp.copy()))
    inputs=np.rint(np.linspace(1,60,13)).astype(int).tolist()
    first_seen=np.full_like(ids,-1);visible_count=np.zeros_like(ids,np.uint8)
    camera_frames=[];object_transforms=[]
    for other in inputs:
        if other==frame:continue
        scene.frame_set(other);deps=bpy.context.evaluated_depsgraph_get();cam=scene.camera;center=cam.matrix_world.translation.copy()
        camera_frames.append(np.array(cam.matrix_world));object_transforms.append(np.array(objects['rigid_moving_sphere'].matrix_world))
        for k,y,x,name,lp in candidates:
            point=objects[name].matrix_world@lp;uv=world_to_camera_view(scene,cam,point)
            if not (0<=uv.x<=1 and 0<=uv.y<=1 and uv.z>0):continue
            hit=scene.ray_cast(deps,center,(point-center).normalized())
            if hit[0] and hit[4].original.name==name and (hit[1]-point).length<=1e-4:
                recover[k,y,x]=True;visible_count[k,y,x]+=1
                if first_seen[k,y,x]<0:first_seen[k,y,x]=other
    scene.frame_set(frame)
    np.savez_compressed(output/'surface-gt-prototype.npz',object_id=ids,polygon_id=faces,object_local_hit=local,
                        known=known,aperture_hidden=hidden,cross_frame_recoverable=recover,aperture_xy=disk,input_frames=inputs,
                        first_visible_other_frame=first_seen,visible_other_frame_count=visible_count,
                        center_depth_axial_m=center_depth,center_object_id=center_ids,center_polygon_id=center_faces,
                        center_object_local_hit=center_local,aperture_depth_axial_m=aperture_depth,
                        intrinsics_opencv=np.array([[fx,0,size/2],[0,fx,size/2],[0,0,1.]]),
                        camera_c2w_blender=np.array(matrix),camera_c2w_opencv=np.array(matrix)@np.diag([1,-1,-1,1]),
                        other_frame_ids=[i for i in inputs if i!=frame],
                        other_camera_c2w_blender=np.stack(camera_frames),other_moving_object_matrices=np.stack(object_transforms),
                        object_names=list(objects),object_ids=[object_ids[n] for n in objects],
                        object_world_matrices=np.stack([np.array(objects[n].matrix_world) for n in objects]))
    record=dict(frame=frame,resolution=[size,size],samples=samples,fstop=fstop,focus_m=focus,input_frames=inputs,
                known_ray_fraction=float(known.mean()),aperture_hidden_image_fraction=float(hidden.any(0).mean()),
                recoverable_image_fraction=float(recover.any(0).mean()),
                tracked_surface_rays=len(candidates),elapsed_seconds=time.monotonic()-started,
                surface_identity='stable object ID + polygon ID + object-local hit point; rigid transform at each frame',
                tolerance_m=1e-4,object_ids=object_ids,depth_convention='axial camera depth in meters; 0 means no hit',
                camera_convention='Blender right/up/back and OpenCV right/down/front; column-vector c2w matrices',
                per_surface_visibility='first visible input frame and number of visible other input frames; checked by point distance, not object ID alone',
                camera_center_max_displacement_m=float(np.max(np.linalg.norm(np.stack(camera_frames)[:,:3,3]-np.array(origin),axis=-1))),
                moving_object_max_displacement_m=float(np.max(np.linalg.norm(np.stack(object_transforms)[:,:3,3]-np.array(objects['rigid_moving_sphere'].matrix_world.translation),axis=-1))),
                invariants=dict(recoverable_subset_of_aperture_hidden=bool(np.all(~recover|hidden)),
                                hidden_subset_of_known=bool(np.all(~hidden|known)),
                                first_seen_consistent=bool(np.all((first_seen>=0)==recover))),
                excludes='out-of-frame center projections and missing first hits; no object-ID-only matching')
    (output/'surface-gt-prototype.json').write_text(json.dumps(record,indent=2)+'\n');print('B1_GT',json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blend',type=Path,help='Existing scene when outputs use a fresh subdirectory')
    p.add_argument('--mode',choices=['build','render','gt'],default='build');p.add_argument('--frames',nargs='+',type=int,default=[1,30])
    p.add_argument('--samples',nargs='+',type=int,default=[16]);p.add_argument('--fstops',nargs='+',type=float,default=[1.4,5.6])
    p.add_argument('--focuses',nargs='+',type=float,default=[5.3,8.4]);p.add_argument('--size',type=int,default=512)
    p.add_argument('--device',choices=['CPU','GPU'],default='CPU');p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--pinhole',action='store_true');a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='build':scene=build_scene(a.output)
    else:bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve() if a.blend else a.output/'b1-procedural.blend'));scene=bpy.context.scene
    if a.mode in ['build','render']:render_sparse(scene,a.output,a.frames,a.samples,a.fstops,a.focuses,a.size,a.device,a.seed,a.pinhole)
    if a.mode=='gt':surface_gt(scene,a.output,size=a.size)


if __name__=='__main__':main()
