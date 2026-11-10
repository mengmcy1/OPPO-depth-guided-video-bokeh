"""Blender B2-only sparse renders and per-object camera-ray peeling.

Jobs are explicit JSON data. No scene layout or material is changed. A hidden
object keeps shadow, glossy, diffuse and transmission ray visibility unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import bpy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);jobs=json.loads(a.jobs.read_text(encoding='utf-8'))
    stop_file=a.jobs.resolve().parent/'STOP-RENDER'
    current=None;baseline={}
    for job in jobs:
        blend=str(Path(job['blend']).resolve())
        if blend!=current:
            bpy.ops.wm.open_mainfile(filepath=blend);current=blend
            baseline={o.name:dict(camera=o.visible_camera,shadow=o.visible_shadow,
                diffuse=o.visible_diffuse,glossy=o.visible_glossy,transmission=o.visible_transmission) for o in bpy.context.scene.objects}
        scene=bpy.context.scene
        for obj in scene.objects:obj.visible_camera=baseline[obj.name]['camera']
        scene.render.threads_mode='FIXED';scene.render.threads=8
        scene.cycles.device='GPU';scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False
        scene.render.use_motion_blur=False;scene.render.resolution_x=scene.render.resolution_y=job.get('size',512);scene.render.resolution_percentage=100
        pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
        for device in pref.devices:device.use=device.type=='OPTIX'
        if not any(d.use and d.type=='OPTIX' for d in pref.devices):raise RuntimeError('No enabled OptiX GPU')
        metadata=json.loads((Path(blend).parent/'scene.json').read_text(encoding='utf-8'))
        id_names={int(o['id']):o['name'] for o in metadata['objects']}
        hidden_ids=sorted(job.get('hidden_object_ids',[]));hidden_names=[id_names[i] for i in hidden_ids]
        if len(hidden_ids)>2:raise ValueError('B2-only permits at most two per-object peeling steps')
        for name in hidden_names:bpy.data.objects[name].visible_camera=False
        for obj in scene.objects:
            for attr in ['shadow','diffuse','glossy','transmission']:
                assert getattr(obj,'visible_'+attr)==baseline[obj.name][attr]
        frames=job['frames'];pinhole=job.get('pinhole',False)
        if not pinhole and len(frames)>5:raise ValueError('Full60-frame DOF is outside this task')
        out=Path(job['output']).resolve();out.mkdir(parents=True,exist_ok=True);summary=out/'timings.json'
        rows=json.loads(summary.read_text(encoding='utf-8')) if summary.exists() else []
        for frame in frames:
            scene.frame_set(frame)
            for stop in job.get('fstops',[1.4]):
                for focus in job.get('focuses',[1.4]):
                    for spp in job['samples']:
                        if stop_file.exists():
                            print('B2_ONLY_STOP_FILE',str(stop_file),flush=True);return
                        seed=int(job['seed']);lens='pinhole' if pinhole else f'N{stop:g}-F{focus:g}'
                        hide='none' if not hidden_ids else '-'.join(map(str,hidden_ids))
                        key=f'f{frame:03d}-{lens}-s{spp}-seed{seed}-hide{hide}'
                        if any(r['key']==key for r in rows):continue
                        if (out/(key+'.exr')).exists():raise FileExistsError('Unrecorded existing output: '+key)
                        scene.camera.data.dof.use_dof=not pinhole;scene.camera.data.dof.aperture_fstop=stop;scene.camera.data.dof.focus_distance=focus
                        scene.cycles.samples=spp;scene.cycles.seed=seed
                        scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='32'
                        scene.render.filepath=str(out/(key+'.exr'));start=time.monotonic();bpy.ops.render.render(write_still=True);seconds=time.monotonic()-start
                        scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8'
                        bpy.data.images['Render Result'].save_render(str(out/(key+'.png')),scene=scene)
                        row=dict(key=key,scene=job['scene'],source_blend=blend,frame=frame,fstop=stop,focus_m=focus,samples=spp,seed=seed,
                            pinhole=pinhole,hidden_object_ids=hidden_ids,hidden_object_names=hidden_names,
                            non_camera_ray_visibility_preserved=True,resolution=[scene.render.resolution_x,scene.render.resolution_y],
                            elapsed_seconds_render_and_EXR_write=seconds,exr_bytes=(out/(key+'.exr')).stat().st_size)
                        rows.append(row);summary.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8');print('B2_ONLY_RENDER',json.dumps(row),flush=True)


if __name__=='__main__':main()
