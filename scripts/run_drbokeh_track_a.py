"""Run the single Dr.Bokeh Track A adapter, preserving preflight diagnostics and outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baselines.drbokeh import OfficialDrBokeh,map_geometry,render_saved,lens_differences


def write_json(path,data):
    with path.open('x') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def save_png(path,value):
    image=np.uint8(np.round(np.clip(value,0,1)*255))
    if path.exists():
        assert np.array_equal(np.asarray(Image.open(path)),image), 'Preserve interrupted output and investigate mismatch'
    else:
        with path.open('xb') as f:Image.fromarray(image).save(f,format='PNG')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/drbokeh-track-a.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/drbokeh-preflight-20260928'))
    p.add_argument('--stage',choices=['preflight','full'],default='preflight')
    p.add_argument('--only',default='')
    p.add_argument('--max-frames',type=int,default=25)
    p.add_argument('--reuse-preflight',type=Path)
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    args.output=(ROOT/args.output).resolve();args.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/args.protocol).read_text())
    repo=ROOT.parent/'DrBokeh'
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==protocol['author_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--','app/Render','app/cuda-src'],text=True)
    old=args.output/'protocol.json'
    if old.exists():assert json.loads(old.read_text())==protocol
    else:write_json(old,protocol)
    clips=ROOT/protocol['input'];cfg=json.loads((clips/'config.json').read_text())
    selected=set(protocol['preflight_scenes']) if args.stage=='preflight' else {s['id'] for s in cfg['scenes']}
    if args.only:
        assert set(args.only.split(','))<=selected
        selected=set(args.only.split(','))
    assert 1<=args.max_frames<=25
    if args.stage=='full':
        assert args.reuse_preflight is not None
        reuse=(ROOT/args.reuse_preflight).resolve()
        acceptance=json.loads((reuse/'acceptance.json').read_text())
        assert acceptance['status']=='PREFLIGHT_PASSED' and acceptance['frozen_protocol']==protocol
        (args.output/'clips').mkdir(exist_ok=True);(args.output/'timing').mkdir(exist_ok=True)
        for sid in protocol['preflight_scenes']:
            for name in ('clips','timing'):
                dst=args.output/name/sid;src=reuse/name/sid
                assert src.exists()
                if dst.exists():assert dst.resolve()==src.resolve()
                else:
                    try:dst.symlink_to(src,target_is_directory=True)
                    except FileExistsError:assert dst.resolve()==src.resolve()
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.manual_seed(0)
    model=OfficialDrBokeh(repo,protocol)
    from app.Render.DScatter.scatter import Multi_Layer_Renderer
    comparison_renderers={l:Multi_Layer_Renderer(l).cuda().eval() for l in (71,101)} if args.stage=='preflight' else {}
    print('OFFICIAL_PIPELINE_LOADED',protocol['author_commit'],'stage',args.stage,flush=True)
    for scene in cfg['scenes']:
        sid=scene['id']
        if sid not in selected:continue
        src=clips/sid;meta=json.loads((src/'metadata.json').read_text())
        out=args.output/'clips'/sid;pred=out/'pred_drbokeh';timing=args.output/'timing'/sid
        pred.mkdir(parents=True,exist_ok=True);timing.mkdir(parents=True,exist_ok=True)
        if (timing/'complete.json').exists():
            assert json.loads((timing/'complete.json').read_text())['frames']==25
            assert len(list(pred.glob('*.png')))==25
            print('REUSE',sid,flush=True);continue
        depths=sorted((src/'disparity').glob('*.png'));assert len(depths)==25
        rows=[]
        for t,path in enumerate(depths[:args.max_frames]):
            record=timing/f'{t:05d}.json'
            if record.exists():
                assert (pred/f'{t:05d}.png').exists()
                rows.append(json.loads(record.read_text()));continue
            assert path.name.startswith(f'{t:05d}_zf_')
            rgb=np.asarray(Image.open(src/'rgb'/f'{t:05d}.png')).astype(np.float32)/255
            d=np.asarray(Image.open(path)).astype(np.float64)/255
            focus=float(path.stem.split('_zf_')[1])
            assert abs(focus-meta['frames'][t]['focus_disparity'])<=5.1e-9
            assert rgb.shape==(576,1024,3) and d.shape==(576,1024)
            dn,fn,kn,geometry=map_geometry(d,focus,scene['K'])
            assert geometry['signed_CoC_error_float64']<=protocol['mapping_tolerance']['float64_max_abs']
            assert geometry['signed_CoC_error_float32']<=protocol['mapping_tolerance']['float32_max_abs']
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
            value,audit,arrays=model.predict(rgb,dn,fn,kn)
            torch.cuda.synchronize()
            audit.update(scene=sid,frame=t,geometry=geometry,seconds=time.monotonic()-start,
                         peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                         input_RGB=str((src/'rgb'/f'{t:05d}.png').relative_to(ROOT)),input_disparity=str(path.relative_to(ROOT)))
            audit['coverage_passed']=all(x['radial_tail_mass_at_max']<=protocol['kernel_check']['omitted_radial_scatter_weight_mass_tolerance'] for x in audit['actual_layers'].values())
            save_png(pred/f'{t:05d}.png',value)
            save_arrays=args.stage=='preflight' or t in protocol['diagnostic_keyframes']
            if save_arrays:
                diagnostic=out/'diagnostics';diagnostic.mkdir(exist_ok=True)
                npz=diagnostic/f'{t:05d}.npz'
                if npz.exists():
                    with np.load(npz) as old_arrays:
                        assert old_arrays.files==list(arrays)
                        assert all(np.array_equal(old_arrays[k],v) for k,v in arrays.items())
                else:
                    with npz.open('xb') as f:np.savez_compressed(f,**arrays)
                audit['diagnostics']=str(npz.relative_to(args.output))
            if args.stage=='preflight' and sid in protocol['layered_lens_check']['scenes']:
                # Reload disk data: both windows receive exactly the same saved post-gamma RGBAD.
                with np.load(npz) as saved:
                    a=render_saved(comparison_renderers[71],saved,kn,fn,protocol['gamma'],protocol['offset'])
                    b=render_saved(comparison_renderers[101],saved,kn,fn,protocol['gamma'],protocol['offset'])
                    assert np.max(np.abs(a-value))<=1e-7, 'Saved-layer rendering disagrees with official full pipeline'
                    audit['lens_comparison']=lens_differences(a,b,saved['ldf_alpha'],saved['input_disparity_normalized'],protocol['layered_lens_check'])
                    low=(saved['layer_0_weight']<1e-8)|(saved['layer_1_weight']<1e-8)
                    audit['lens_comparison']['low_weight_fraction']=float(low.mean())
                    audit['lens_comparison']['max_abs_at_low_weight']=float(np.abs(a-b)[low].max()) if low.any() else 0.
                dest=out/'lens101';dest.mkdir(exist_ok=True);save_png(dest/f'{t:05d}.png',b)
            if args.stage=='full':assert audit['coverage_passed'], 'Actual repaired layer exceeds declared coverage; preserve frame and inspect'
            write_json(record,audit);rows.append(audit)
            print('FRAME',sid,t,'seconds',round(audit['seconds'],3),'coverage',audit['coverage_passed'],flush=True)
        if len(rows)==25:
            write_json(timing/'complete.json',dict(scene=sid,frames=25,seconds=sum(x['seconds'] for x in rows),
                peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in rows),all_coverage_passed=all(x['coverage_passed'] for x in rows),
                mask_threshold_empty_frames=sum(x['LDF_mask']['threshold_empty'] for x in rows),
                mask_threshold_full_frames=sum(x['LDF_mask']['threshold_full'] for x in rows)))
            print('DONE',sid,flush=True)


if __name__=='__main__':main()
