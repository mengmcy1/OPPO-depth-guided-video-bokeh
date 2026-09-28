"""Blender background entry: match the pilot's planar textures, camera and linear-light colors."""
import bpy,sys,json,time,math,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--samples',type=int,default=1024);p.add_argument('--fstop-scale',type=float,default=None);p.add_argument('--only',default='');p.add_argument('--frames',type=int,default=0);p.add_argument('--pinhole',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--seed-offset',type=int,default=0);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
if a.fstop_scale is None:
 settings=a.input/'render_settings.json'
 c=json.loads((settings if settings.exists() else a.input/'config.json').read_text())
 a.fstop_scale=c.get('fstop_scale',c.get('cycles_fstop_scale',1.))
preferences=bpy.context.preferences.addons['cycles'].preferences
preferences.compute_device_type='OPTIX';preferences.get_devices()
devices=[]
for d in preferences.devices:
 d.use=d.type=='OPTIX'
 if d.use:devices.append(d.name)
assert devices,'No OPTIX GPU device'
print('GPU_DEVICES',devices,flush=True)
for folder in sorted(a.input.glob('*')):
 if not (folder/'metadata.json').exists() or (a.only and folder.name!=a.only):continue
 meta=json.loads((folder/'metadata.json').read_text());c=meta['config'];W,H=c['width'],c['height']
 bpy.ops.wm.read_factory_settings(use_empty=True)
 preferences=bpy.context.preferences.addons['cycles'].preferences
 preferences.compute_device_type='OPTIX';preferences.get_devices()
 for d in preferences.devices:d.use=d.type=='OPTIX'
 assert preferences.compute_device_type=='OPTIX' and any(d.use and d.type=='OPTIX' for d in preferences.devices)
 print('ACTIVE_RENDER_DEVICES',[(d.name,d.type,d.use) for d in preferences.devices],flush=True)
 s=bpy.context.scene
 s.render.engine='CYCLES';s.cycles.device='GPU';s.cycles.samples=a.samples;s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False;s.cycles.transparent_max_bounces=32
 s.render.resolution_x=W;s.render.resolution_y=H;s.render.resolution_percentage=100;s.render.film_transparent=False
 s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.render.image_settings.color_depth='16'
 s.view_settings.view_transform='Standard';s.view_settings.look='None';s.view_settings.exposure=0;s.view_settings.gamma=1
 s.cycles.pixel_filter_type='BOX';s.cycles.filter_width=1.
 world=bpy.data.worlds.new('Black');world.use_nodes=True;world.node_tree.nodes.get('Background').inputs['Color'].default_value=(0,0,0,1);s.world=world
 camera=bpy.data.cameras.new('MatchedCamera');obj=bpy.data.objects.new('Camera',camera);s.collection.objects.link(obj);s.camera=obj
 camera.lens=c['focal_length_mm'];camera.sensor_width=c['sensor_width_mm'];camera.sensor_fit='HORIZONTAL';camera.dof.use_dof=not a.pinhole
 f=camera.lens/1000;sensor=camera.sensor_width/1000
 camera.dof.aperture_fstop=f*f*W/(2*sensor*c['K'])*a.fstop_scale
 camera.dof.aperture_blades=0
 materials=[]
 def plane(name,z,file,padding=0,transparent=False):
  width=(W+2*padding)/W*sensor/f*z;height=(H+2*padding)/W*sensor/f*z
  mesh=bpy.data.meshes.new(name);mesh.from_pydata([(-width/2,-height/2,-z),(width/2,-height/2,-z),(width/2,height/2,-z),(-width/2,height/2,-z)],[],[(0,1,2,3)])
  uv=mesh.uv_layers.new()
  for i,xy in enumerate([(0,0),(1,0),(1,1),(0,1)]):uv.data[i].uv=xy
  obj=bpy.data.objects.new(name,mesh);s.collection.objects.link(obj)
  mat=bpy.data.materials.new(name);mat.use_nodes=True;n=mat.node_tree.nodes;n.clear();links=mat.node_tree.links
  tex=n.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(file),check_existing=True);tex.image.alpha_mode='STRAIGHT';tex.interpolation='Closest';tex.extension='EXTEND'
  em=n.new('ShaderNodeEmission');links.new(tex.outputs['Color'],em.inputs['Color']);out=n.new('ShaderNodeOutputMaterial')
  if transparent:
   trans=n.new('ShaderNodeBsdfTransparent');mix=n.new('ShaderNodeMixShader');links.new(tex.outputs['Alpha'],mix.inputs[0]);links.new(trans.outputs[0],mix.inputs[1]);links.new(em.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],out.inputs['Surface'])
  else:links.new(em.outputs[0],out.inputs['Surface'])
  mesh.materials.append(mat)
  return tex
 plane('Background',c['background_depth_m'],folder/'layers/background.png',64)
 first=meta['frames'][0]
 for li,(file,z) in enumerate(zip(first['foreground_files'],first['foreground_depths_m'])):materials.append(plane('Foreground'+str(li),z,folder/file,0,True))
 tag='pinhole_optix' if a.pinhole else f'cycles_optix_s{a.samples}_f{a.fstop_scale:g}'
 tag+=f'_seed{a.seed_offset}' if a.seed_offset else ''
 out=folder/tag;out.mkdir(exist_ok=a.resume)
 times=json.loads((out/'timing.json').read_text())['seconds'] if a.resume and (out/'timing.json').exists() else []
 for frame in meta['frames'][:a.frames or None]:
  if a.resume and (out/f"{frame['index']:05d}.png").exists():continue
  # Public-generator-style reproduction varies axial depth per frame while
  # preserving image-space layer placement. Uniform XYZ scale preserves projection.
  bg_scale=frame.get('background_depth_m',c['background_depth_m'])/c['background_depth_m']
  bpy.data.objects['Background'].scale=(bg_scale,)*3
  for li,(z,z0) in enumerate(zip(frame['foreground_depths_m'],first['foreground_depths_m'])):
   bpy.data.objects['Foreground'+str(li)].scale=(z/z0,)*3
  for tex,file in zip(materials,frame['foreground_files']):
   old=tex.image;tex.image=bpy.data.images.load(str(folder/file),check_existing=True)
   if old.users==0:bpy.data.images.remove(old)
  camera.dof.focus_distance=frame['focus_distance_m'];s.cycles.seed=c['seed']+frame['index']+a.seed_offset;s.render.filepath=str(out/f"{frame['index']:05d}.png")
  start=time.perf_counter();bpy.ops.render.render(write_still=True);times.append(time.perf_counter()-start)
 (out/'timing.json').write_text(json.dumps({'seconds':times,'total_seconds':sum(times),'samples':a.samples,'aperture_fstop':camera.dof.aperture_fstop,'gpu':devices,'compute_device_type':preferences.compute_device_type,'blender':bpy.app.version_string,'per_frame_layer_depths':True,'scope':'render + PNG IO; includes first scene update; no denoising'},indent=2))
 print('SCENE_DONE',folder.name,tag,sum(times),flush=True)
