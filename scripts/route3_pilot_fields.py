"""Pilot sampling, native data-pass and alpha-compositing correctness checks."""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
CONFIG=Path(os.environ.get('ROUTE3_PILOT_CONFIG',ROOT/'configs/route3-network-pilot-2026-11-12.json'))
SETTINGS=json.loads(CONFIG.read_text());OUT=Path(SETTINGS['output']);OUT=OUT if OUT.is_absolute() else ROOT/OUT
RECORDS=ROOT/'03_实验记录/records';PREFIX=SETTINGS.get('record_prefix','route3-network-pilot-2026-11-12')
from route3_upper_bound_evaluate import display,score
from route3_ldr_u1r import model


def read(path):
    v=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if v is None:raise FileNotFoundError(path)
    return v[..., [2,1,0,3]] if v.ndim==3 and v.shape[-1]==4 else v[...,::-1].copy() if v.ndim==3 else v


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def header(path):
    """Read EXR attributes without decoding channels or modifying float IDs."""
    def string(f):
        b=bytearray()
        while True:
            c=f.read(1)
            if not c or c==b'\0':return b.decode()
            b.extend(c)
    attrs={}
    with path.open('rb') as f:
        assert f.read(4)==b'v/1\x01';f.read(4)
        while True:
            name=string(f)
            if not name:break
            kind=string(f);size=struct.unpack('<I',f.read(4))[0];data=f.read(size)
            attrs[name]=dict(type=kind,bytes=size)
            if kind=='string':attrs[name]['value']=data.decode(errors='replace')
    return attrs


def noise():
    torch.set_num_threads(6);net=model();rows=[]
    for label in ['pilot-thin','pilot-multilayer']:
        p=OUT/label/'preflight'
        for focus in [1.4,5.]:
            a=read(p/f'N1.4-F{focus}-s1024-seed20261112.exr')[...,:3];b=read(p/f'N1.4-F{focus}-s1024-seed20261113.exr')[...,:3];low=read(p/f'N1.4-F{focus}-s512-seed20261112.exr')[...,:3]
            masks=dict(full_image=np.ones((512,512),bool));s=score(display(a),display(b),masks,net)['full_image'];c=score(display(a),display(low),masks,net)['full_image']
            rows.append(dict(scene=label,focus_m=focus,fstop=1.4,spp=1024,independent_noise=s,same_seed_512_difference=c,linear_noise_RMSE=float(np.sqrt(np.mean((a.astype(np.float64)-b)**2))),selected_spp=1024,scope='pilot configuration;N2 inherited conservatively from largerN1.4aperture',noise_supports_0p01_LPIPS=bool(3*s['patch_lpips']<.01)))
    write(RECORDS/(PREFIX+'-sampling.json'),dict(rows=rows,all_noise_supports_0p01_LPIPS=all(r['noise_supports_0p01_LPIPS'] for r in rows),not_old_B2_formal_criteria=True,denoising=False,motion_blur=False,filter_type='BOX',filter_width=1.,LPIPS='0.1.4 Alex0.1;64pxpatches'))


def check():
    results=json.loads((OUT/'center-data-jobs-results.json').read_text())
    results+=json.loads((OUT/'aperture-corrected-jobs-results.json').read_text());rows=[]
    for label in ['pilot-thin','pilot-multilayer']:
        folder=OUT/label;metadata=json.loads((folder/'scene.json').read_text());geo=np.load(folder/'scene-geometry.npz');center=[];aperture=[];cached_path=None;cached=None
        for job in results:
            if job['scene']!=label or not job.get('data_passes'):continue
            pos=read(job['passes']['Position']);ids=read(job['passes']['IndexOB'])[...,0].round().astype(np.int16);valid=ids>0
            if 'aperture_xy' in job:
                case=f'f{job["frame"]:03d}-N{job["fstop"]:g}-F{job["focus"]:g}-K16';k=int(job['id'].split('-camera-fixed')[0].rsplit('k',1)[-1]);path=folder/'geometry'/(case+'.npz')
                if cached_path!=path:
                    with np.load(path) as archive:cached={key:archive[key] for key in ['world_position','object_id']}
                    cached_path=path
                expected=cached['world_position'][k];obj=cached['object_id'][k]
            else:
                g=np.load(folder/'geometry'/f'f{job["frame"]:03d}-center.npz');expected=g['world_position'];obj=g['object_id']
            same=valid&(ids==obj)&np.isfinite(expected).all(-1);dist=np.linalg.norm(pos-expected,axis=-1);stats=dict(id=job['id'],pixels=int(valid.sum()),object_ID_agreement=float((ids[valid]==obj[valid]).mean()),world_error_same_ID_p50_p99_max_m=np.quantile(dist[same],[.5,.99,1]).tolist(),world_error_same_ID_gt1mm_fraction=float((dist[same]>.001).mean()))
            if 'aperture_xy' in job:aperture.append(stats)
            else:
                local=read(job['passes']['ObjectLocal']);stats['local_error_same_ID_p99_m']=float(np.quantile(np.linalg.norm(local-g['local_surface_point'],axis=-1)[same],.99));crypto=Path(job['passes']['Cryptomatte']);attrs=header(crypto);stats['cryptomatte']=dict(path=str(crypto),header_fields=list(attrs),manifest_present=any('manifest' in k for k in attrs));center.append(stats)
        full=read(folder/'inputs/f030.exr')[...,:3];fg=read(folder/'preflight/foreground.exr');bg=read(folder/'preflight/background.exr')[...,:3];alpha=fg[...,3];pred=fg[...,:3]+(1-alpha[...,None])*bg;error=pred-full;edge=(alpha>0)&(alpha<1);inside=(alpha==0)|(alpha==1)
        alpha_check=dict(formula='Ihat=F_premultiplied+(1-alpha)*B;F=Fpremul/alpha foralpha>0',all_linear_RMSE=float(np.sqrt(np.mean(error.astype(np.float64)**2))),edge_linear_RMSE=float(np.sqrt(np.mean(error[edge].astype(np.float64)**2))) if edge.any() else None,interior_linear_RMSE=float(np.sqrt(np.mean(error[inside].astype(np.float64)**2))),abs_error_p50_p95_p99_max=np.quantile(abs(error),[.5,.95,.99,1]).tolist(),edge_pixels=int(edge.sum()),camera_visibility_only_removal=True,shadow_reflection_indirect_unchanged=True,noise_and_subpixel_background_covariance_limit_exact_pixelwise_identity=True)
        # Explicit input RGBs for the frozen inference components.
        for frame in geo['input_frames']:
            source=read(folder/'inputs'/f'f{frame:03d}.exr')[...,:3];rgb=display(source.clip(0,1));cv2.imwrite(str(folder/'inputs'/f'f{frame:03d}.png'),np.uint8(np.rint(rgb[...,::-1]*255)))
        rows.append(dict(scene=label,center_native_passes=center,aperture_native_passes=aperture,alpha_reconstruction=alpha_check,expected_inputs=13,expected_aperture_passes=192,static_geometry=metadata['static_object_transforms_verified'],no_dynamic_objects=len(geo['dynamic_object_ids'])==0))
        write(folder/'field-checks.json',rows[-1])
    write(RECORDS/(PREFIX+'-fields.json'),dict(completed=True,rows=rows,Blender='4.5.3 LTS',color_filter='BOX1.0',data_filter='native unfilteredPosition/Z/ObjectIndex;1sppforoffaxis',data_Embree_crosscheck='exportedsameBlendergeometry;nativepassesaretraininglabels',initial_animated_camera_passes_invalid_and_excluded=True,corrected_unanimated_camera=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['noise','check'],required=True);a=p.parse_args();globals()[a.mode]()


if __name__=='__main__':main()
