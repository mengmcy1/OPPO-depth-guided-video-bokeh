"""Run the fixed input-permission diagnostic; preserve the original 200-scene result."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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
    branches=['gt_alpha','oracle_layers'] if args.branch=='both' else [args.branch]
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
            records={branch:run/branch/'timing'/sid/f'{t:05d}.json' for branch in branches}
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
        for branch in branches:
            timing=run/branch/'timing'/sid
            rows=[json.loads(p.read_text()) for p in sorted(timing.glob('[0-9]*.json'))]
            if len(rows)==25 and not (timing/'complete.json').exists():
                write_json(timing/'complete.json',dict(scene=sid,frames=25,seconds=sum(r['seconds'] for r in rows)))
                print('DONE',branch,sid,flush=True)


def means(rows):
    return {key:float(np.mean([r[key] for r in rows])) for key in KEYS}


def clip_metrics(config,base,run,branch,workers):
    """Schedule disjoint scene evaluations; preserve the existing per-clip metric implementation."""
    dest=run/branch;report=dest/'metrics.json'
    if report.exists():return
    parts=dest/'metrics-parts';parts.mkdir(exist_ok=True)
    prediction_root=str((dest/'clips').relative_to(ROOT))
    ids=config['scene_ids'];seen=set();saved=[]
    for path in sorted(parts.glob('*.json')):
        data=json.loads(path.read_text())
        assert data['prediction_root']==prediction_root and data['prediction_dir']=='pred_drbokeh'
        chosen=[r['scene'] for r in data['rows']]
        assert len(set(chosen))==len(chosen) and set(chosen)<=set(ids) and not seen.intersection(chosen)
        seen.update(chosen);saved.append(data)
    pending=[sid for sid in ids if sid not in seen]
    logs=ROOT/config['logs'];stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    def worker(index,chosen):
        path=parts/f'{stamp}-part{index}.json'
        command=['scripts/evaluate_renderer_pilot.py','--input',base['input'],'--only',','.join(chosen),
            '--reference-dir','gt_mpib_public_srgb','--reference-label','MPIB-public proxy',
            '--prediction-root',prediction_root,'--prediction-dir','pred_drbokeh','--author-rm','--torchmetrics',
            '--scope',config['name']+'; input-permission recovery, not an automatically selected main-table baseline','--report',str(path)]
        with (logs/f'{branch}-metrics-{stamp}-part{index}.log').open('x') as f:
            subprocess.run([sys.executable,*command],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        data=json.loads(path.read_text());assert [r['scene'] for r in data['rows']]==chosen
        return data
    shards=[pending[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,i,s) for i,s in enumerate(shards) if s]
        for future in futures:saved.append(future.result())
    assert saved
    template=saved[0]
    for data in saved:
        assert data['metrics']==template['metrics'] and data['scope']==template['scope']
        assert data['prediction_root']==prediction_root and data['prediction_dir']=='pred_drbokeh'
    rows=[r for data in saved for r in data['rows']]
    assert len(rows)==len(ids) and {r['scene'] for r in rows}==set(ids)
    by_id={r['scene']:r for r in rows};template['rows']=[by_id[sid] for sid in ids]
    for label,values in template['mean'].items():
        for key in values:
            vals=[r[key] for r in rows if r['renderer']==label];assert np.isfinite(vals).all()
            # Match the original manifest order and float64 aggregation.
            template['mean'][label][key]=float(np.mean([r[key] for r in template['rows'] if r['renderer']==label]))
    template['scheduling']='Disjoint scene processes; identical metric functions; manifest-order float64 aggregation; no distribution metrics split into chunks.'
    write_json(report,template)
    print('CLIP_METRICS_COMPLETE',branch,len(rows),flush=True)


def evaluation(config,base,run,metric_workers=1):
    """Reuse old predictions/features, compute all branches on the same manifest."""
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
    logs=ROOT/config.get('logs','logs/drbokeh-protocol-recovery-v2-20260928');logs.mkdir(exist_ok=True)
    def execute(command,name):
        with (logs/name).open('a') as f:
            f.write('\nATTEMPT '+datetime.now(timezone.utc).isoformat()+'\n');f.flush()
            subprocess.run([sys.executable,*command],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    for branch in ('gt_alpha','oracle_layers'):
        dest=run/branch;metrics=dest/'metrics.json'
        prediction_root=str((dest/'clips').relative_to(ROOT))
        if metric_workers>1:clip_metrics(config,base,run,branch,metric_workers)
        if not metrics.exists():
            execute(['scripts/evaluate_renderer_pilot.py','--input',base['input'],'--only',','.join(ids),
                '--reference-dir','gt_mpib_public_srgb','--reference-label','MPIB-public proxy',
                '--prediction-root',prediction_root,'--prediction-dir','pred_drbokeh','--author-rm','--torchmetrics',
                '--scope',config['name']+'; input-permission recovery, not an automatically selected main-table baseline','--report',str(metrics)],branch+'-metrics.log')
        data=json.loads(metrics.read_text());assert [r['scene'] for r in data['rows']]==ids
        assert data['prediction_root']==prediction_root and data['prediction_dir']=='pred_drbokeh'
        comparison[branch]=dict(rows=data['rows'],mean=means(data['rows']),distributions={})
        for phase in ('vfid','fvd'):
            execute(['scripts/evaluate_author_public_metrics.py','--clips',base['input'],'--only',','.join(ids),
                     '--prediction-root',prediction_root,'--prediction-dir','pred_drbokeh',
                     '--reference-features',str(cache.relative_to(ROOT)),'--output',str((dest/'author-public').relative_to(ROOT)),
                     '--phase',phase],branch+'-'+phase+'.log')
            score=json.loads((dest/'author-public'/f'{phase}.json').read_text())
            assert score['protocol']['scene_ids']==ids and score['video_count']==len(ids)
            comparison[branch]['distributions'][phase]=next(iter(score['values'].values()))
            filename='fvd-videomae-features.npz' if phase=='fvd' else 'vfid-i-features.npz'
            features=np.load(dest/'author-public'/filename)
            assert np.array_equal(features['gt'],np.load(cache/filename)['gt']) and np.isfinite(features['pred']).all()
    for branch,result in comparison.items():
        assert all(np.isfinite(v) for v in result['mean'].values())
        result['groups']={group:dict(n=len(chosen),mean=means([r for r in result['rows'] if r['scene'] in chosen]))
                          for group,chosen in [('original_preflight_8',config['original_preflight']),('posthoc_failures_3',config['posthoc_failure_cases'])]}
        manifest=json.loads((ROOT/base['input']/'config.json').read_text())
        K={s['id']:s['K'] for s in manifest['scenes']}
        result['by_K']={str(k):dict(n=sum(K[r['scene']]==k for r in result['rows']),
            mean=means([r for r in result['rows'] if K[r['scene']]==k])) for k in (4,8,16,32)}
        if len(ids)==200:
            remaining=set(ids)-set(config['original_preflight'])-set(config['posthoc_failure_cases'])
            result['groups']['remaining_189']=dict(n=len(remaining),mean=means([r for r in result['rows'] if r['scene'] in remaining]))
            m=result['mean'];dist=result['distributions']
            result['paper_comparison']=dict(PSNR=m['PSNR_TorchMetrics'],SSIM=m['SSIM_TorchMetrics'],FD=m['FD'],RM=m['RM'],FVD=dist['fvd'],VFID_I=dist['vfid'])
            result['signed_difference_to_paper']={k:v-base['paper_DrBokeh'][k] for k,v in result['paper_comparison'].items()}
        for sid in ids:
            paths=sorted((run/branch/'clips'/sid/'pred_drbokeh').glob('*.png'))
            assert [p.name for p in paths]==[f'{t:05d}.png' for t in range(25)]
            for p in paths:
                with Image.open(p) as im:assert im.size==(1024,576) and im.mode=='RGB';im.verify()
    status='FULL_200_INPUT_PROTOCOL_RECOVERY_COMPLETE_REVIEW_PENDING' if len(ids)==200 else 'FIXED_11_INPUT_DIAGNOSTIC_METRICS_COMPLETE_VISUAL_REVIEW_PENDING'
    write_json(ROOT/config['report'],dict(status=status,
        completed_utc=datetime.now(timezone.utc).isoformat(),config=config,base_protocol=base,
        comparison=comparison,limits=config['limits']))
    print('DRBOKEH_PROTOCOL_DIAGNOSTIC_COMPLETE',config['report'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/drbokeh-protocol-recovery-v2.json')
    p.add_argument('--phase',choices=['all','inference','evaluate'],default='all')
    p.add_argument('--only',default='');p.add_argument('--max-frames',type=int,default=25)
    p.add_argument('--branch',choices=['both','gt_alpha','oracle_layers'],default='both')
    p.add_argument('--workers',type=int,default=1,help='Disjoint scene processes within each input branch; branches remain sequential')
    p.add_argument('--metric-workers',type=int,default=1,help='Disjoint scene processes for per-clip metrics only; FVD/VFID remain whole-set')
    args=p.parse_args();os.chdir(ROOT)
    config=json.loads((ROOT/args.config).read_text());base=json.loads((ROOT/config['base_protocol']).read_text())
    for key in ('gamma','highlight','offset','lens'):assert base[key]==config['fixed'][key]
    assert 1<=args.max_frames<=25
    assert 1<=args.workers<=4
    assert 1<=args.metric_workers<=4
    expected=config.get('scene_count',11)
    assert expected in (11,200) and len(config['scene_ids'])==len(set(config['scene_ids']))==expected
    run=ROOT/config['output'];run.mkdir(parents=True,exist_ok=True)
    protocol=dict(config=config,base_protocol=base)
    if (run/'protocol.json').exists():assert json.loads((run/'protocol.json').read_text())==protocol
    else:write_json(run/'protocol.json',protocol)
    if expected==200:
        manifest=json.loads((ROOT/base['input']/'config.json').read_text())
        assert config['scene_ids']==[s['id'] for s in manifest['scenes']]
        # Reused diagnostic scenes can be omitted from every inference shard.
        # Register their original full-public outputs before final integrity checks too.
        for sid in config['scene_ids']:
            src=ROOT/config['baseline_output']/'clips'/sid
            dst=run/'full_public/clips'/sid
            assert src.is_dir()
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():assert dst.resolve()==src.resolve()
            else:dst.symlink_to(src,target_is_directory=True)
        review=json.loads((ROOT/config['visual_review']).read_text())
        assert review['status']=='FIXED_55_VISUAL_REVIEW_COMPLETE_NO_NEW_IMPLEMENTATION_BLOCKER'
        source=ROOT/config['reuse_diagnostic'];previous=json.loads((source/'protocol.json').read_text())
        assert previous['base_protocol']==base
        for key in ('branches','fixed','oracle_disparity','asset_checks'):
            assert previous['config'][key]==config[key], 'Cannot reuse frames across changed input protocols'
        assert review['scene_ids']==previous['config']['scene_ids']
        for sid in previous['config']['scene_ids']:
            for branch in ('gt_alpha','oracle_layers'):
                assert json.loads((source/branch/'timing'/sid/'complete.json').read_text())['frames']==25
                assert len(list((source/branch/'clips'/sid/'pred_drbokeh').glob('*.png')))==25
                for category in ('clips','timing'):
                    dst=run/branch/category/sid;src=source/branch/category/sid;dst.parent.mkdir(parents=True,exist_ok=True)
                    if dst.exists():assert dst.resolve()==src.resolve()
                    else:dst.symlink_to(src,target_is_directory=True)
    if args.phase=='inference':inference(args,config,base,run)
    elif args.phase=='evaluate':evaluation(config,base,run,args.metric_workers)
    else:
        assert not args.only and args.max_frames==25 and args.branch=='both'
        # Separate processes release LDF/LaMa/CUDA allocations before metric extraction.
        stages=[('inference','gt_alpha'),('inference','oracle_layers'),('evaluate','both')] if expected==200 else [('inference','both'),('evaluate','both')]
        for phase,branch in stages:
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',args.config,'--phase',phase,'--branch',branch,'--metric-workers',str(args.metric_workers)]
            if phase!='inference' or args.workers==1:
                subprocess.run(command,check=True,cwd=ROOT)
                continue
            assert expected==200 and branch!='both'
            pending=[sid for sid in config['scene_ids'] if not (run/branch/'timing'/sid/'complete.json').exists()]
            if not pending:
                print('BRANCH_ALREADY_COMPLETE',branch,flush=True);continue
            count=min(args.workers,len(pending))
            shards=[pending[i::count] for i in range(count)]
            assert len({sid for shard in shards for sid in shard})==sum(map(len,shards))==len(pending)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            logs=ROOT/config.get('logs','logs/drbokeh-protocol-recovery-v2-20260928')/('parallel-'+stamp)
            logs.mkdir(parents=True)
            workers=[];jobs=[]
            for i,shard in enumerate(shards):
                logfile=logs/f'{branch}-worker{i}.log'
                with logfile.open('x') as f:
                    child=subprocess.Popen(command+['--only',','.join(shard)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                workers.append(child);jobs.append(dict(pid=child.pid,scenes=shard,log=str(logfile.relative_to(ROOT))))
            write_json(run/f'scheduling-{stamp}.json',dict(branch=branch,supervisor=os.getpid(),workers=jobs,
                scope='Scene scheduling only; unchanged protocol, each scene has exactly one producer; no standardized efficiency claim'))
            print('PARALLEL_BRANCH_START',branch,'workers',count,'scenes',len(pending),'logs',str(logs.relative_to(ROOT)),flush=True)
            codes=[child.wait() for child in workers]
            assert all(code==0 for code in codes), ('Worker failed; completed outputs preserved; inspect worker logs',codes)
            assert all((run/branch/'timing'/sid/'complete.json').exists() for sid in config['scene_ids'])
            print('PARALLEL_BRANCH_COMPLETE',branch,flush=True)


if __name__=='__main__':main()
