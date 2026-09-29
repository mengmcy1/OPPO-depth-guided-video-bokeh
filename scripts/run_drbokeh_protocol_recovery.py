"""Run the fixed input-permission diagnostic; preserve the original 200-scene result."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
KEYS=['PSNR_TorchMetrics','SSIM_TorchMetrics','LPIPS','FD','RM','RM_author_public_2transitions']


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def save_png(path,value):
    encoded=np.uint8(np.round(np.clip(value,0,1)*255))
    if path.exists():
        assert np.array_equal(np.asarray(Image.open(path)),encoded), 'Interrupted PNG differs; preserve and inspect'
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:Image.fromarray(encoded).save(f,format='PNG')


def known_layers(src,frame,rgb,d,alpha,geometry,gamma):
    """Recover the already saved assets, not a newly synthesized scene."""
    bg=np.asarray(Image.open(src/'layers/background.png').convert('RGB'))[64:-64,64:-64].astype(np.float32)/255
    assert bg.shape==rgb.shape
    rgba=[np.asarray(Image.open(src/p).convert('RGBA')).astype(np.float32)/255 for p in frame['foreground_files']]
    depths=frame['foreground_depths_m']+[frame['background_depth_m']]
    assert len(rgba)==4 and all(a.shape==(*rgb.shape[:2],4) for a in rgba)
    assert depths==sorted(depths), 'Renderer requires front-to-back order'
    # Use float32 exactly as the existing scene generator when auditing PNGs.
    disp=np.full(rgb.shape[:2],1/depths[-1],np.float32)
    union=np.zeros(rgb.shape[:2],np.float32);composite=bg.copy()
    for a,z in reversed(list(zip(rgba,depths[:-1]))):
        aa=a[...,3];composite=a[...,:3]*aa[...,None]+composite*(1-aa[...,None])
        disp=np.where(aa>.5,1/z,disp)
        union=aa+union*(1-aa)
    encode=lambda a:np.uint8(np.round(np.clip(a,0,1)*255))
    assert np.array_equal(encode(composite),encode(rgb)), 'Saved assets do not reconstruct RGB'
    assert np.array_equal(encode(union),encode(alpha[...,0])), 'Union-alpha mismatch'
    assert np.array_equal(encode(disp),encode(d)), 'Visible disparity mismatch'
    planes=[float(encode(np.array(1/z,dtype=np.float32)))/255 for z in depths]
    # Check quantized per-plane assembly independently of the float32 depth audit.
    check=np.full(d.shape,planes[-1])
    for a,plane in reversed(list(zip(rgba,planes[:-1]))):check=np.where(a[...,3]>.5,plane,check)
    assert np.array_equal(check,d), 'Quantized plane convention differs from canonical input'
    rgba.append(np.concatenate([bg,np.ones((*bg.shape[:2],1),np.float32)],axis=-1))
    layers=[]
    for a,plane in zip(rgba,planes):
        layer=np.concatenate([a.astype(np.float64),np.full((*d.shape,1),(plane-geometry['min'])/geometry['span'])],axis=-1)
        layer[...,:3]=layer[...,:3]**gamma
        layers.append(layer)
    return layers,dict(RGB_PNG_exact=True,union_alpha_PNG_exact=True,visible_disparity_PNG_exact=True,
                       quantized_plane_disparities=planes,source_files=frame['foreground_files'],background='layers/background.png')


def inference(args,config,base,run):
    import cv2
    import torch
    from baselines.drbokeh import OfficialDrBokeh,map_geometry,array_stats,tail_mass
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    repo=ROOT.parent/'DrBokeh'
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==base['author_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--','app/Render','app/cuda-src'],text=True)
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.manual_seed(0)
    model=OfficialDrBokeh(repo,base)
    from app.Render.DScatter.scatter import Multi_Layer_Renderer
    renderer=Multi_Layer_Renderer(base['lens'],gpu_occlusion=True).cuda().eval()
    clips=ROOT/base['input'];scenes={s['id']:s for s in json.loads((clips/'config.json').read_text())['scenes']}
    selected=args.only.split(',') if args.only else config['scene_ids']
    assert set(selected)<=set(config['scene_ids'])
    for sid in config['scene_ids']:
        if sid not in selected:continue
        src=clips/sid;meta=json.loads((src/'metadata.json').read_text())
        paths=sorted((src/'disparity').glob('*.png'));assert len(paths)==25
        baseline=ROOT/config['baseline_output']/'clips'/sid
        dst=run/'full_public/clips'/sid;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert dst.resolve()==baseline.resolve()
        else:dst.symlink_to(baseline,target_is_directory=True)
        for t,path in enumerate(paths[:args.max_frames]):
            records={branch:run/branch/'timing'/sid/f'{t:05d}.json' for branch in ('gt_alpha','oracle_layers')}
            if all(p.exists() for p in records.values()):
                for branch in records:assert (run/branch/'clips'/sid/'pred_drbokeh'/f'{t:05d}.png').exists()
                continue
            assert path.name.startswith(f'{t:05d}_zf_')
            rgb=np.asarray(Image.open(src/'rgb'/f'{t:05d}.png')).astype(np.float32)/255
            d=np.asarray(Image.open(path)).astype(np.float64)/255
            alpha=np.asarray(Image.open(src/'alpha'/f'{t:05d}.png')).astype(np.float32)[...,None]/255
            focus=float(path.stem.split('_zf_')[1]);assert abs(focus-meta['frames'][t]['focus_disparity'])<=5.1e-9
            dn,fn,kn,geometry=map_geometry(d,focus,scenes[sid]['K'])
            for dtype in ('float64','float32'):
                assert geometry['signed_CoC_error_'+dtype]<=base['mapping_tolerance'][dtype+'_max_abs']
            layers,assets=known_layers(src,meta['frames'][t],rgb,d,alpha,geometry,base['gamma'])
            for branch,record in records.items():
                if record.exists():continue
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                if branch=='gt_alpha':
                    value,audit,arrays=model.predict(rgb,dn,fn,kn,alpha_override=alpha)
                else:
                    with torch.no_grad():value=renderer.inference(layers,kn,fn)**(1/base['gamma'])-base['offset']
                    assert value.shape==rgb.shape and np.isfinite(value).all()
                    audit=dict(calls=dict(LDF=0,LaMa=0,MiDaS=0,layer_renderer=5),actual_layers={},
                               output=dict(**array_stats(value),clipped_fraction=float(((value<0)|(value>1)).mean())))
                    for i,layer in enumerate(layers):
                        radius=kn*np.abs(layer[...,4]-fn)
                        audit['actual_layers'][str(i)]=dict(disparity_range=[float(layer[...,4].min()),float(layer[...,4].max())],
                            actual_radius_max=float(radius.max()),radial_tail_mass_at_max=tail_mass(float(radius.max()),base['lens']))
                    arrays={}
                torch.cuda.synchronize()
                audit.update(scene=sid,frame=t,branch=branch,geometry=geometry,assets=assets,
                             seconds=time.monotonic()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
                audit['coverage_passed']=all(x['radial_tail_mass_at_max']<=base['kernel_check']['omitted_radial_scatter_weight_mass_tolerance'] for x in audit['actual_layers'].values())
                assert audit['coverage_passed'], 'Preserve inputs; kernel coverage failed; no automatic lens change'
                out=run/branch/'clips'/sid
                save_png(out/'pred_drbokeh'/f'{t:05d}.png',value)
                if arrays and t in base['diagnostic_keyframes']:
                    diagnostic=out/'diagnostics'/f'{t:05d}.npz';diagnostic.parent.mkdir(exist_ok=True)
                    if not diagnostic.exists():
                        with diagnostic.open('xb') as f:np.savez_compressed(f,**arrays)
                    audit['diagnostics']=str(diagnostic.relative_to(run))
                write_json(record,audit)
                print('FRAME',branch,sid,t,'seconds',round(audit['seconds'],3),flush=True)
        for branch in ('gt_alpha','oracle_layers'):
            timing=run/branch/'timing'/sid
            rows=[json.loads(p.read_text()) for p in sorted(timing.glob('[0-9]*.json'))]
            if len(rows)==25 and not (timing/'complete.json').exists():
                write_json(timing/'complete.json',dict(scene=sid,frames=25,seconds=sum(r['seconds'] for r in rows)))
                print('DONE',branch,sid,flush=True)


def means(rows):
    return {key:float(np.mean([r[key] for r in rows])) for key in KEYS}


def evaluation(config,base,run):
    """Reuse old predictions/features, recompute distribution metrics on SAME 11 scenes."""
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT.parent/'any-to-bokeh'))
    sys.path.insert(0,str(ROOT.parent/'content-debiased-fvd'))
    from cdfvd.fvd import cdfvd
    from cdfvd.utils.metric_utils import FeatureStats
    from metrics.fid_metrics.fid import calculate_fid
    ids=config['scene_ids'];baseline=ROOT/config['baseline_output']
    for branch in ('gt_alpha','oracle_layers'):
        for sid in ids:assert json.loads((run/branch/'timing'/sid/'complete.json').read_text())['frames']==25
    cache=run/'reference-features';cache.mkdir(exist_ok=True)
    original=ROOT/'outputs/author-public-metrics-20260928'
    protocol=json.loads((original/'protocol.json').read_text());protocol['scene_ids']=ids
    if (cache/'protocol.json').exists():assert json.loads((cache/'protocol.json').read_text())==protocol
    else:write_json(cache/'protocol.json',protocol)
    old=json.loads((ROOT/config['baseline_record']).read_text());rows=[r for r in old['rows'] if r['scene'] in ids]
    assert [r['scene'] for r in rows]==ids
    comparison={'full_public':dict(rows=rows,mean=means(rows),distributions={})}
    # Reuse 200-scene features but NOT the 200-scene distribution score.
    for phase,filename,windows in [('fvd','fvd-videomae-features.npz',1),('vfid','vfid-i-features.npz',2)]:
        original_features=np.load(original/filename);saved=np.load(baseline/'author-public'/filename)
        assert list(original_features['scene_ids'])==list(saved['scene_ids'])
        assert np.array_equal(original_features['gt'],saved['gt'])
        positions=[list(saved['scene_ids']).index(s) for s in ids]
        indices=[windows*i+j for i in positions for j in range(windows)]
        gt,pred=saved['gt'][indices],saved['pred'][indices]
        path=cache/filename
        if path.exists():
            prior=np.load(path);assert list(prior['scene_ids'])==ids and np.array_equal(prior['gt'],gt)
        else:
            with path.open('xb') as f:np.savez(f,gt=gt,scene_ids=ids)
        if phase=='fvd':
            real,fake=[FeatureStats(capture_mean_cov=True,capture_all=True) for _ in range(2)]
            real.append(gt);fake.append(pred)
            value=float(cdfvd.__new__(cdfvd).compute_fvd_from_stats(fake_stats=fake,real_stats=real))
        else:value=float(calculate_fid(gt,pred))
        assert np.isfinite(value)
        comparison['full_public']['distributions'][phase]=value
    logs=ROOT/'logs/drbokeh-protocol-recovery-v2-20260928';logs.mkdir(exist_ok=True)
    def execute(command,name):
        with (logs/name).open('a') as f:
            f.write('\nATTEMPT '+datetime.now(timezone.utc).isoformat()+'\n');f.flush()
            subprocess.run([sys.executable,*command],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    for branch in ('gt_alpha','oracle_layers'):
        dest=run/branch;metrics=dest/'metrics.json'
        prediction_root=str((dest/'clips').relative_to(ROOT))
        if not metrics.exists():
            execute(['scripts/evaluate_renderer_pilot.py','--input',base['input'],'--only',','.join(ids),
                '--reference-dir','gt_mpib_public_srgb','--reference-label','MPIB-public proxy',
                '--prediction-root',prediction_root,'--prediction-dir','pred_drbokeh','--author-rm','--torchmetrics',
                '--scope','Selected 11-scene input-permission diagnostic; not a new main-table baseline','--report',str(metrics)],branch+'-metrics.log')
        data=json.loads(metrics.read_text());assert [r['scene'] for r in data['rows']]==ids
        assert data['prediction_root']==prediction_root and data['prediction_dir']=='pred_drbokeh'
        comparison[branch]=dict(rows=data['rows'],mean=means(data['rows']),distributions={})
        for phase in ('vfid','fvd'):
            execute(['scripts/evaluate_author_public_metrics.py','--clips',base['input'],'--only',','.join(ids),
                     '--prediction-root',prediction_root,'--prediction-dir','pred_drbokeh',
                     '--reference-features',str(cache.relative_to(ROOT)),'--output',str((dest/'author-public').relative_to(ROOT)),
                     '--phase',phase],branch+'-'+phase+'.log')
            score=json.loads((dest/'author-public'/f'{phase}.json').read_text())
            assert score['protocol']['scene_ids']==ids and score['video_count']==11
            comparison[branch]['distributions'][phase]=next(iter(score['values'].values()))
            filename='fvd-videomae-features.npz' if phase=='fvd' else 'vfid-i-features.npz'
            features=np.load(dest/'author-public'/filename)
            assert np.array_equal(features['gt'],np.load(cache/filename)['gt']) and np.isfinite(features['pred']).all()
    for branch,result in comparison.items():
        assert all(np.isfinite(v) for v in result['mean'].values())
        result['groups']={group:dict(n=len(chosen),mean=means([r for r in result['rows'] if r['scene'] in chosen]))
                          for group,chosen in [('original_preflight_8',config['original_preflight']),('posthoc_failures_3',config['posthoc_failure_cases'])]}
        for sid in ids:
            paths=sorted((run/branch/'clips'/sid/'pred_drbokeh').glob('*.png'))
            assert [p.name for p in paths]==[f'{t:05d}.png' for t in range(25)]
            for p in paths:
                with Image.open(p) as im:assert im.size==(1024,576) and im.mode=='RGB';im.verify()
    write_json(ROOT/config['report'],dict(status='FIXED_11_INPUT_DIAGNOSTIC_METRICS_COMPLETE_VISUAL_REVIEW_PENDING',
        completed_utc=datetime.now(timezone.utc).isoformat(),config=config,base_protocol=base,
        comparison=comparison,limits=config['limits']))
    print('DRBOKEH_PROTOCOL_DIAGNOSTIC_COMPLETE',config['report'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/drbokeh-protocol-recovery-v2.json')
    p.add_argument('--phase',choices=['all','inference','evaluate'],default='all')
    p.add_argument('--only',default='');p.add_argument('--max-frames',type=int,default=25)
    args=p.parse_args();os.chdir(ROOT)
    config=json.loads((ROOT/args.config).read_text());base=json.loads((ROOT/config['base_protocol']).read_text())
    for key in ('gamma','highlight','offset','lens'):assert base[key]==config['fixed'][key]
    assert 1<=args.max_frames<=25
    assert len(config['scene_ids'])==11 and len(set(config['scene_ids']))==11
    run=ROOT/config['output'];run.mkdir(parents=True,exist_ok=True)
    protocol=dict(config=config,base_protocol=base)
    if (run/'protocol.json').exists():assert json.loads((run/'protocol.json').read_text())==protocol
    else:write_json(run/'protocol.json',protocol)
    if args.phase=='inference':inference(args,config,base,run)
    elif args.phase=='evaluate':evaluation(config,base,run)
    else:
        assert not args.only and args.max_frames==25
        # Separate processes release LDF/LaMa/CUDA allocations before metric extraction.
        for phase in ('inference','evaluate'):
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--config',args.config,'--phase',phase],check=True,cwd=ROOT)


if __name__=='__main__':main()
