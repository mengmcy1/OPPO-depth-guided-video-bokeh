"""Frozen B2-only LaMa lookup, BokehMe O3 and Oracle-input Dr.Bokeh.

Run fill/drbokeh in the existing magicbokeh environment and bokehme in
dofgaussian. This script produces predictions only, never evaluation metrics.
All modes require the frozen criteria and a parent-granted GPU window.
Manifest item lens scalars override the shared frame NPZ defaults. Raw no-hit
depth stays0 in center_depth_raw; center_depth uses the common10000m method
approximation, with center_surface_valid excluding sky from GT surface claims.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def caches(base):
    locations=dict(TMPDIR='tmp',TMP='tmp',TEMP='tmp',HF_HOME='hf',TORCH_HOME='torch',
        XDG_CACHE_HOME='xdg',MPLCONFIGDIR='matplotlib',TORCH_EXTENSIONS_DIR='torch_extensions',
        CUPY_CACHE_DIR='cupy',CUDA_CACHE_PATH='cuda',PIP_CACHE_DIR='pip')
    for key,folder in locations.items():
        p=base/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ['PYTHONUTF8']='1'
    return {key:os.environ[key] for key in locations}


def linear(rgb):
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4).astype(np.float32)


def shared_input(data):
    rgb=np.asarray(data['rgb_srgb'],np.float32)
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
        raise ValueError('All methods require the same display RGB[0,1] input')
    derived=linear(rgb)
    if 'visible_rgb_linear' in data:
        supplied=np.asarray(data['visible_rgb_linear'],np.float32)
        if supplied.shape!=rgb.shape or not np.isfinite(supplied).all() or np.max(abs(supplied-derived))>2e-6:
            raise ValueError('3D visible RGB differs from the shared LDR-derived input; ask parent to confirm policy')
    alpha_key='foreground_alpha' if 'foreground_alpha' in data else 'GTforeground_alpha'
    alpha=np.asarray(data[alpha_key],np.float32)
    if alpha.shape!=rgb.shape[:2] or not np.isfinite(alpha).all() or alpha.min()<0 or alpha.max()>1:
        raise ValueError('GT foreground alpha must be HxW pixel coverage in[0,1]')
    return rgb,derived,alpha


def depth_protocol(data):
    """Known mesh hits plus the shared 10km method-only sky approximation."""
    depth=np.asarray(data['center_depth'],np.float32)
    if not np.isfinite(depth).all() or (depth<=0).any():
        raise ValueError('Method center depth must be finite positive, including common far-field sky')
    if 'center_surface_valid' in data:
        known=np.asarray(data['center_surface_valid'],bool)
        source='center_surface_valid'
    elif 'center_depth_raw' in data:
        raw=np.asarray(data['center_depth_raw'],np.float32)
        known=np.isfinite(raw)&(raw>0);source='center_depth_raw finite positive'
    else:
        known=np.ones(depth.shape,bool);source='all finite positive provided depth; no unknown-sky metadata'
    if known.shape!=depth.shape:raise ValueError('center surface validity shape mismatch')
    if 'center_depth_raw' in data:
        raw=np.asarray(data['center_depth_raw'],np.float32)
        if raw.shape!=depth.shape or not np.isfinite(raw).all() or (raw<0).any():
            raise ValueError('Raw depth must be finite nonnegative, with0 denoting no hit')
        if np.any(known&(raw<=0)) or np.any((~known)&(raw!=0)):
            raise ValueError('Raw depth and center surface validity disagree')
        if not np.allclose(depth[known],raw[known],rtol=1e-6,atol=1e-6):
            raise ValueError('Method depth must preserve known first-hit surface depths')
    if (~known).any() and not np.allclose(depth[~known],1e4,rtol=0,atol=1e-3):
        raise ValueError('No-hit sky must use the common method-only 10000m far-field depth')
    return depth,known,dict(surface_validity_source=source,known_surface_pixels=int(known.sum()),
        unknown_sky_pixels=int((~known).sum()),known_surface_fraction=float(known.mean()),
        shared_far_field_depth_m=10000.,
        unknown_sky_policy='1/z uses10000m, never zero defocus; not a GT surface or recoverable target region')


def physical_control(data,entry=None):
    from src.c1_bokehme import pipeline_defocus
    depth,_,_=depth_protocol(data)
    entry=entry or {}
    focus=float(entry['focus_distance_m'] if 'focus_distance_m' in entry else data['focus_distance_m'])
    aperture=float(entry['aperture_radius_m'] if 'aperture_radius_m' in entry else data['aperture_radius_m'])
    if not np.isfinite(focus) or focus<=0 or not np.isfinite(aperture) or aperture<0:
        raise ValueError('Finite positive focus and nonnegative aperture radius required')
    K=np.asarray(data['K'],np.float32)
    control,valid=pipeline_defocus(depth,float(K[0,0]),aperture,focus)
    assert valid.all()
    return depth,K,focus,aperture,control


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--criteria',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--mode',choices=['fill','bokehme','drbokeh'],required=True)
    p.add_argument('--repository',type=Path,help='Defaults to existing BokehMe or DrBokeh checkout')
    p.add_argument('--cache',type=Path,default=Path('/root/autodl-tmp/route3-cache'))
    a=p.parse_args();a.manifest=a.manifest.resolve();a.criteria=a.criteria.resolve();a.output=a.output.resolve();a.record=a.record.resolve()
    cache_paths=caches(a.cache.resolve())
    criteria=json.loads(a.criteria.read_text(encoding='utf-8'))
    if criteria.get('frozen') is not True:raise ValueError('B2 criteria must explicitly contain frozen:true before any method output')
    manifest=json.loads(a.manifest.read_text(encoding='utf-8'));entries=manifest['items']
    for entry in entries:
        if entry.get('dataset','').lower()!='blender' or entry['scene'] not in ['B2-low','B2-high']:
            raise ValueError('This entry point only accepts the authorized Blender B2 scenes')
        if Path(entry['id']).name!=entry['id']:raise ValueError('id must be a single directory name')
        if not entry.get('uses_gt_foreground_alpha'):raise ValueError('Declare GT foreground alpha input explicitly')
    if len({e['id'] for e in entries})!=len(entries):raise ValueError('Manifest ids must be unique')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Set the parent-granted GPU explicitly to CUDA_VISIBLE_DEVICES=0')
    subprocess.run(['nvidia-smi'],check=True)
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(4)
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    a.output.mkdir(parents=True,exist_ok=False)
    repository=(a.repository or ROOT.parent/('BokehMe' if a.mode=='bokehme' else 'DrBokeh')).resolve()
    original_cwd=Path.cwd()
    if a.mode=='fill':
        os.chdir(repository);sys.path.insert(0,str(repository))
        spec=importlib.util.spec_from_file_location('b2_lama_rgb',repository/'app/Render/rgb_inpainting.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=module.RGB_Inpainting_Inference()
    elif a.mode=='bokehme':
        from src.baselines.bokehme import setup,predict
        pipeline,settings,nets,counts=setup(repository,ROOT/'models/BokehMe',dict(gamma=2.2,highlight=False,defocus_scale=10.))
    else:
        from src.baselines.drbokeh import OfficialDrBokeh,map_geometry
        protocol=json.loads((ROOT/'configs/drbokeh-track-a-frozen.json').read_text(encoding='utf-8'))
        protocol['lens']=int(manifest.get('drbokeh_lens',protocol['lens']))
        protocol['oracle_alpha']=True;protocol['oracle_depth']=True
        model=OfficialDrBokeh(repository,protocol,instrument=True)
    results=[]
    for entry in entries:
        source=Path(entry['input_npz']);source=source if source.is_absolute() else original_cwd/source
        source=source.resolve();data=np.load(source)
        rgb,visible_linear,alpha=shared_input(data)
        out=a.output/entry['id'];out.mkdir()
        row=dict(id=entry['id'],scene=entry['scene'],frame_id=entry['frame_id'],input_npz=str(source),
            mode=a.mode,GT_foreground_alpha=True,fill_mask_uses_GT_segmentation=entry.get('fill_mask_uses_gt_foreground_segmentation',False),
            fill_uses_true_hidden_depth=entry.get('fill_uses_true_hidden_depth',False),geometry_oracle_setting=entry.get('geometry_oracle_setting',False),
            shared_input='same sRGB[0,1] and its true-linear inverse transfer for all methods',input_RGB_range=[float(rgb.min()),float(rgb.max())],
            input_metadata=entry)
        if 'center_depth' in data:
            _,known_surface,depth_info=depth_protocol(data)
            row['center_depth_protocol']=depth_info
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        if a.mode=='fill':
            mask=np.asarray(data['fill_mask'],bool)
            if mask.shape!=alpha.shape:raise ValueError('LaMa mask shape mismatch')
            result=model.inference(rgb,mask[...,None].astype(np.float32))
            if result is None or not np.isfinite(result).all():raise RuntimeError('LaMa did not produce finite fill colors')
            fill=np.where(mask[...,None],np.clip(result,0,1),rgb).astype(np.float32)
            np.savez_compressed(out/'fill-image.npz',fill_rgb_srgb=fill,fill_rgb_linear=linear(fill),fill_mask=mask)
            cv2.imwrite(str(out/'fill-image.png'),np.uint8(fill[...,::-1]*255))
            row.update(fill_mask_pixels=int(mask.sum()),raw_LaMa_range=[float(result.min()),float(result.max())],
                outside_mask_max_error=float(abs(fill[~mask]-rgb[~mask]).max()) if (~mask).any() else None,
                lookup_file=str(out/'fill-image.npz'),role='C_fill lookup only: multiply remaining ray transmittance after joint visible+hidden depth-sorted alpha composition')
        elif a.mode=='bokehme':
            if not entry.get('uses_gt_depth'):raise ValueError('O3 must receive declared GT depth')
            depth,K,focus,aperture,control=physical_control(data,entry)
            prediction,stats=predict(pipeline,settings,nets,np.ascontiguousarray(rgb),np.ascontiguousarray(control))
            prediction=np.clip(prediction,0,1).astype(np.float32)
            np.savez_compressed(out/'prediction.npz',prediction_srgb=prediction,O3_srgb=prediction,physical_signed_radius=control*10,
                                center_surface_valid=known_surface)
            cv2.imwrite(str(out/'O3.png'),np.uint8(prediction[...,::-1]*255))
            row.update(method='O3 BokehMe GT depth',focus_distance_m=focus,aperture_radius_m=aperture,
                       gamma=2.2,defocus_scale=10.,physical_control='fx*a*(1/z-1/F)/10',
                       signed_CoC_radius_range=[float(control.min()*10),float(control.max()*10)],stats=stats)
        else:
            if not entry.get('uses_gt_depth'):raise ValueError('Oracle-input Dr.Bokeh must receive declared GT depth')
            depth,K,focus,aperture,_=physical_control(data,entry)
            normalized,focus_n,strength_n,mapping=map_geometry(1/depth.astype(np.float64),1/focus,float(K[0,0])*aperture)
            prediction,stats,arrays=model.predict(rgb,normalized,focus_n,strength_n,alpha_override=alpha[...,None])
            bounded=np.clip(prediction,0,1).astype(np.float32)
            np.savez_compressed(out/'prediction.npz',prediction_srgb=bounded,DrBokeh_GT_srgb=bounded,
                unbounded_rgb=prediction.astype(np.float32),GT_alpha=alpha,center_depth_method=depth,center_surface_valid=known_surface,
                **{k:v for k,v in arrays.items() if k in ['fg_rgbad_render','bg_rgbad_render']})
            cv2.imwrite(str(out/'Oracle-input-DrBokeh.png'),np.uint8(bounded[...,::-1]*255))
            row.update(method='Oracle-input Dr.Bokeh',focus_distance_m=focus,aperture_radius_m=aperture,
                physical_mapping=mapping,diagnostics=stats,lens=protocol['lens'],
                kernel_note='Official +1 radius soft-kernel and fixed support retained; no score-based CoC adjustment',
                optional_GT_hidden_layer='not run by explicit scope')
        torch.cuda.synchronize();row['seconds']=time.monotonic()-started;row['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        save_json(a.output/'runtime.json',dict(mode=a.mode,criteria=str(a.criteria),rows=results,completed=False))
    record=dict(mode=a.mode,criteria=str(a.criteria),manifest=str(a.manifest),rows=results,completed=True,
        repository=str(repository),interpreter=sys.executable,torch=torch.__version__,caches=cache_paths,
        model_training=False,normal_inference_DrBokeh=False,CamVFX_experiment=False,
        extra_HDR_given_to_3D=False,model_parameter_counts=counts if a.mode=='bokehme' else None,
        timing_boundary='Per-item model calls, diagnostic statistics and NPZ/PNG output writes; excludes model setup/loading and inputNPZ reads',
        color_policy='sRGB[0,1] shared input; 3D true-linear integration; official2D gamma2.2; quality evaluation elsewhere clips displayRGB[0,1]')
    save_json(a.record,record);save_json(a.output/'runtime.json',record)


if __name__=='__main__':main()
