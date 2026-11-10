"""Authorized eight-frame L1 temporal Cycles extension, no new scene."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import bpy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    jobs=json.loads(a.jobs.read_text());subprocess.run(['nvidia-smi'],check=True);rows=[]
    for j in jobs:
        assert j['scene']=='B2-high' and set(j['frames'])<=set(range(27,35)) and len(j['frames'])<=8
        bpy.ops.wm.open_mainfile(filepath=j['blend']);scene=bpy.context.scene
        scene.render.threads_mode='FIXED';scene.render.threads=8;scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100
        scene.render.use_motion_blur=False;scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False;scene.cycles.device='GPU'
        pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
        for d in pref.devices:d.use=d.type=='OPTIX'
        if not any(d.use and d.type=='OPTIX' for d in pref.devices):raise RuntimeError('No OptiX device')
        out=Path(j['output']);out.mkdir(parents=True,exist_ok=False)
        scene.camera.data.dof.use_dof=True;scene.camera.data.dof.aperture_fstop=1.4;scene.camera.data.dof.focus_distance=5.
        scene.cycles.samples=1024;scene.cycles.seed=20261110
        scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='32'
        for frame in j['frames']:
            scene.frame_set(frame);path=out/f'f{frame:03d}-N1.4-F5-s1024-seed20261110.exr'
            if path.exists():raise FileExistsError(path)
            scene.render.filepath=str(path);start=time.monotonic();bpy.ops.render.render(write_still=True)
            rows.append(dict(frame=frame,EXR=str(path),seconds=time.monotonic()-start,samples=1024,seed=20261110))
        (out/'timings.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
