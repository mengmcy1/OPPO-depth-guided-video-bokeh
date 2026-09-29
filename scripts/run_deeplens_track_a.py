"""Frozen native-TF1 DeepLens inference, preflight checks, and existing-metric evaluation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT.parent/'envs/deeplens-tf1-cpu/bin/python'
METRIC_PY=Path('/root/miniconda3/envs/magicbokeh/bin/python')


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def save_png(path,bgr):
    import cv2
    encoded=np.uint8(np.round(np.clip(bgr,0,1)*255))
    if path.exists():assert np.array_equal(cv2.imread(str(path)),encoded)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        ok,buffer=cv2.imencode('.png',encoded);assert ok
        with path.open('xb') as f:f.write(buffer.tobytes())


def native_model(config):
    import cv2
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('TF_ENABLE_ONEDNN_OPTS')=='0'
    sys.path.insert(0,str(ROOT/'src'))
    from baselines.deeplens import DeepLensExternal,NATIVE_TF1
    assert NATIVE_TF1
    cv2.setNumThreads(2)
    model=DeepLensExternal(ROOT.parent/'DeepLens',ROOT/config['checkpoint'],threads=config['threads_per_worker'])
    assert model.audit['runtime']=='1.15.0' and model.audit['restored_tensors']==210
    return model


def worker(config,out,selected):
    import cv2
    model=native_model(config)
    from baselines.deeplens import map_geometry
    write(out/'runtime'/f'worker-{os.getpid()}.json',model.audit)
    manifest=json.loads((ROOT/config['input']/'config.json').read_text())
    for scene in manifest['scenes']:
        sid=scene['id']
        if sid not in selected:continue
        src=ROOT/config['input']/sid;dest=out/'clips'/sid/'pred_deeplens';timing=out/'timing'/sid
        timing.mkdir(parents=True,exist_ok=True)
        if (timing/'complete.json').exists():
            assert len(list(dest.glob('*.png')))==25;continue
        metadata=json.loads((src/'metadata.json').read_text());paths=sorted((src/'disparity').glob('*.png'))
        assert len(paths)==25
        for t,path in enumerate(paths):
            record=timing/f'{t:05d}.json';png=dest/f'{t:05d}.png'
            if record.exists():assert png.exists();continue
            assert path.name.startswith(f'{t:05d}_zf_')
            focus=float(path.stem.split('_zf_')[1]);assert abs(focus-metadata['frames'][t]['focus_disparity'])<=5.1e-9
            high=cv2.imread(str(src/'rgb'/f'{t:05d}.png')).astype(np.float32)/255
            raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);assert raw.dtype==np.uint8
            d=raw.astype(np.float64)/255
            assert high.shape==(576,1024,3) and d.shape==(576,1024)
            low=cv2.resize(high,(512,288),interpolation=cv2.INTER_AREA)
            dn,fn,aperture,geometry=map_geometry(d,focus,scene['K'],config['response_scale'])
            start=time.monotonic();value,audit=model.predict(low,high,dn,fn,aperture);seconds=time.monotonic()-start
            save_png(png,value)
            write(record,dict(scene=sid,frame=t,seconds=seconds,geometry=geometry,output=audit,
                input_RGB=str((src/'rgb'/f'{t:05d}.png').relative_to(ROOT)),input_disparity=str(path.relative_to(ROOT))))
            print('FRAME',sid,t,round(seconds,3),'saturated',geometry['saturated'],flush=True)
        records=[json.loads((timing/f'{t:05d}.json').read_text()) for t in range(25)]
        write(timing/'complete.json',dict(scene=sid,frames=25,seconds=sum(r['seconds'] for r in records)))
        print('DONE',sid,flush=True)


def controls(config,out):
    import cv2
    model=native_model(config)
    from baselines.deeplens import map_geometry
    from calibrate_deeplens_control import edge_response
    result=out/'control-check.json'
    if result.exists():return
    rows=[];plateaus=[]
    factor=json.loads((ROOT/config['calibration']).read_text())['disk_edge_width_per_radius']
    for orientation in ['vertical','horizontal']:
        high=np.full((576,1024,3),.25,np.float32)
        if orientation=='vertical':high[:,512:]=.75
        else:high[288:]=.75
        low=cv2.resize(high,(512,288),interpolation=cv2.INTER_AREA)
        for sign in [-1,1]:
            d=np.full((576,1024),(sign+1)/2,np.float64);d[:16,:16]=1-d[0,0]
            focus=(1-sign)/2;values={};widths=[]
            for k in [4,8,16,32]:
                dn,fn,ap,g=map_geometry(d,focus,k,config['response_scale'])
                value,audit=model.predict(low,high,dn,fn,ap);response=edge_response(value,orientation)
                radius=response['edge_10_90_width_px']/factor
                widths.append(response['edge_10_90_width_px']);values[k]=value
                rows.append(dict(orientation=orientation,sign=sign,K=k,geometry=g,output=audit,
                    edge_width_px=response['edge_10_90_width_px'],disk_edge_equivalent_radius_px=radius))
                save_png(out/'controls'/f'{orientation}-{sign}-K{k}.png',value)
            assert widths[0]<widths[1]<widths[2] and widths[2]==widths[3]
            assert np.array_equal(values[16],values[32])
            plateaus.append(dict(orientation=orientation,sign=sign,K16_K32_float_bitwise_equal=True))
    write(result,dict(status='CONTROL_MAPPING_AND_SATURATION_PASSED',rows=rows,plateaus=plateaus,
        scope='Same RGB and geometry acrossK; remote16x16 depth anchor avoids zero span, away from measured edge. Edge-equivalent radii are descriptive, not physical PSF proofs. No benchmark scores.'))
    print('CONTROL_CHECK_COMPLETE',flush=True)


def check(config,out,ids,stage):
    from PIL import Image
    rows=[]
    for sid in ids:
        assert json.loads((out/'timing'/sid/'complete.json').read_text())['frames']==25
        paths=sorted((out/'clips'/sid/'pred_deeplens').glob('*.png'))
        assert [p.name for p in paths]==[f'{t:05d}.png' for t in range(25)]
        for t,p in enumerate(paths):
            with Image.open(p) as im:assert im.size==(1024,576) and im.mode=='RGB';im.verify()
            r=json.loads((out/'timing'/sid/f'{t:05d}.json').read_text());g=r['geometry'];o=r['output']
            assert r['scene']==sid and r['frame']==t and g['span']>0 and 0<=g['aperture']<=1
            assert g['aperture_required']==g['K']*g['span']/config['response_scale']
            assert g['aperture']==min(g['aperture_required'],1) and g['saturated']==(g['aperture_required']>1)
            assert g['signed_equivalence_error_float64']<1e-12 and o['signed_input_max_abs_error']<=1e-6
            rows.append(r)
    byk={}
    for k in [4,8,16,32]:
        group=[r for r in rows if r['geometry']['K']==k]
        byk[str(k)]=dict(frames=len(group),saturated_frames=sum(r['geometry']['saturated'] for r in group),
            saturation_fraction=float(np.mean([r['geometry']['saturated'] for r in group])),
            aperture_required_range=[min(r['geometry']['aperture_required'] for r in group),max(r['geometry']['aperture_required'] for r in group)],
            achieved_aperture_range=[min(r['geometry']['aperture'] for r in group),max(r['geometry']['aperture'] for r in group)])
    summary=dict(status='NUMERICAL_AND_PNG_CHECKS_PASSED',stage=stage,scene_count=len(ids),frames=len(rows),by_K=byk,
        max_signed_input_error=max(r['output']['signed_input_max_abs_error'] for r in rows),
        focus_prime_range=[min(r['geometry']['focus_prime'] for r in rows),max(r['geometry']['focus_prime'] for r in rows)],
        max_output_clip_fraction=max(r['output']['output_clip_fraction'] for r in rows),
        visual_review_complete=False if stage=='preflight' else None)
    if not (out/'checks.json').exists():write(out/'checks.json',summary)
    print('CHECKS_COMPLETE',summary,flush=True)


def evaluate(config,out,ids):
    import fcntl
    # The full supervisor calls this again after inference. One owner evaluates;
    # the second invocation waits and then reuses the finished report.
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        report=ROOT/config['report']
        if report.exists():
            saved=json.loads(report.read_text());assert saved['protocol']==config and len(saved['rows'])==200
            print('DEEPLENS_METRICS_ALREADY_COMPLETE',flush=True);return
        evaluate_locked(config,out,ids)


def evaluate_locked(config,out,ids):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv'],check=True)
    logs=ROOT/config['logs'];parts=out/'metrics';parts.mkdir(exist_ok=True)
    prediction_root=str((out/'clips').relative_to(ROOT))
    def execute(command,name):
        with (logs/name).open('a') as f:
            f.write('\nATTEMPT '+datetime.now(timezone.utc).isoformat()+'\n');f.flush()
            subprocess.run([str(METRIC_PY),*command],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    def wait_for(paths):
        start=time.monotonic()
        while not all(p.exists() for p in paths):
            owner=json.loads((out/'server-jobs.json').read_text())['pid']
            stat=Path('/proc')/str(owner)/'stat'
            if not stat.exists() or stat.read_text().split()[2]=='Z':
                raise RuntimeError('Inference supervisor exited before required completed inputs; preserve partial metrics')
            if time.monotonic()-start>28800:raise TimeoutError('Required input/check not ready after8h')
            time.sleep(10)
    def metric_part(i):
        chosen=ids[i*10:(i+1)*10];path=parts/f'batch{i:02d}.json'
        wait_for([out/'timing'/sid/'complete.json' for sid in chosen])
        if not path.exists():
            execute(['scripts/evaluate_renderer_pilot.py','--input',config['input'],'--only',','.join(chosen),
                '--reference-dir','gt_mpib_public_srgb','--reference-label','MPIB-public proxy',
                '--prediction-root',prediction_root,'--prediction-dir','pred_deeplens','--author-rm','--torchmetrics',
                '--scope',config['name']+'; proxy reproduction, not private author adapter','--report',str(path)],f'metrics-batch{i:02d}.log')
        data=json.loads(path.read_text());assert [r['scene'] for r in data['rows']]==chosen
        assert data['prediction_root']==prediction_root and data['prediction_dir']=='pred_deeplens'
        print('METRIC_BATCH_COMPLETE',i,chosen[0],chosen[-1],flush=True)
        return data['rows']
    # One metric process uses GPU while eight native CPU workers produce scenes.
    # Keep the original per-clip metric implementation and manifest order.
    chunks=[metric_part(i) for i in range(20)]
    allrows=[r for chunk in chunks for r in chunk];assert len(allrows)==200
    index={r['scene']:r for r in allrows};assert set(index)==set(ids);rows=[index[s] for s in ids]
    for phase in ['vfid','fvd']:
        execute(['scripts/evaluate_author_public_metrics.py','--clips',config['input'],
            '--prediction-root',prediction_root,'--prediction-dir','pred_deeplens',
            '--reference-features','outputs/author-public-metrics-20260928','--output',str((out/'author-public').relative_to(ROOT)),
            '--phase',phase],phase+'.log')
    keys=['PSNR_TorchMetrics','SSIM_TorchMetrics','LPIPS','FD','RM','RM_author_public_2transitions']
    mean={k:float(np.mean([r[k] for r in rows])) for k in keys};assert np.isfinite(list(mean.values())).all()
    scenes={s['id']:s for s in json.loads((ROOT/config['input']/'config.json').read_text())['scenes']}
    for row in rows:row['K']=scenes[row['scene']]['K']
    byk={str(k):dict(n=sum(r['K']==k for r in rows),mean={m:float(np.mean([r[m] for r in rows if r['K']==k])) for m in keys}) for k in [4,8,16,32]}
    distributions={}
    for phase,filename,shape in [('vfid','vfid-i-features.npz',(400,1024)),('fvd','fvd-videomae-features.npz',(200,1408))]:
        data=json.loads((out/'author-public'/f'{phase}.json').read_text())
        assert data['video_count']==200 and data['protocol']['scene_ids']==ids
        features=np.load(out/'author-public'/filename);reference=np.load(ROOT/'outputs/author-public-metrics-20260928'/filename)
        assert features['pred'].shape==shape and list(features['scene_ids'])==ids
        assert np.isfinite(features['pred']).all() and np.array_equal(features['gt'],reference['gt'])
        distributions[phase]=next(iter(data['values'].values()))
    compare=dict(PSNR=mean['PSNR_TorchMetrics'],SSIM=mean['SSIM_TorchMetrics'],LPIPS=mean['LPIPS'],FD=mean['FD'],RM=mean['RM'],FVD=distributions['fvd'],VFID_I=distributions['vfid'])
    wait_for([out/'checks.json'])
    write(ROOT/config['report'],dict(status='FULL_200_DEEPLENS_METRICS_COMPLETE_REVIEW_PENDING',completed_utc=datetime.now(timezone.utc).isoformat(),
        protocol=config,mean=mean,by_K=byk,rows=rows,distributions=distributions,paper_comparison=compare,
        signed_difference_to_paper={k:v-config['paper'][k] for k,v in compare.items()},
        verification=json.loads((out/'checks.json').read_text()),limits=config['limits']))
    print('DEEPLENS_200_METRICS_COMPLETE',flush=True)


def preview(config,out,ids):
    from PIL import Image,ImageDraw
    dest=out/'visual-review';dest.mkdir(exist_ok=True)
    for sid in ids:
        page=Image.new('RGB',(1280,1020),'white');draw=ImageDraw.Draw(page)
        src=ROOT/config['input']/sid
        for row,t in enumerate(config['keyframes']):
            info=json.loads((out/'timing'/sid/f'{t:05d}.json').read_text());g=info['geometry']
            d=np.asarray(Image.open(ROOT/info['input_disparity']),dtype=np.float64)/255
            coc=Image.fromarray(np.uint8(np.clip(np.abs(d-g['focus'])*g['K']/32,0,1)*255)).convert('RGB')
            images=[Image.open(src/'rgb'/f'{t:05d}.png').convert('RGB'),Image.open(src/'gt_mpib_public_srgb'/f'{t:05d}.png').convert('RGB'),
                Image.open(out/'clips'/sid/'pred_deeplens'/f'{t:05d}.png').convert('RGB'),coc]
            labels=['AIF','MPIB-public GT',f"DeepLens a={g['aperture']:.3f} sat={g['saturated']}",'target radius /32 (black=focus)']
            for col,(im,label) in enumerate(zip(images,labels)):
                page.paste(im.resize((320,180),Image.Resampling.LANCZOS),(col*320,row*204+24))
                draw.text((col*320+3,row*204+3),f'{sid}/{t:02d} K{g["K"]} {label}',fill='black')
        with (dest/f'{sid}.jpg').open('xb') as f:page.save(f,format='JPEG',quality=95)
        print('PREVIEW',sid,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/deeplens-track-a-frozen.json')
    p.add_argument('--phase',choices=['preflight','full','worker','controls','evaluate','preview'],required=True)
    p.add_argument('--stage',choices=['preflight','full'],default='preflight')
    p.add_argument('--only',default='');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();os.chdir(ROOT);config=json.loads((ROOT/args.config).read_text())
    stage=args.phase if args.phase in ['preflight','full'] else args.stage
    out=ROOT/config['preflight_output' if stage=='preflight' else 'output'];out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==config
    else:write(out/'protocol.json',config)
    manifest=json.loads((ROOT/config['input']/'config.json').read_text())
    ids=config['preflight_scenes'] if stage=='preflight' else [s['id'] for s in manifest['scenes']]
    if args.phase=='preview':
        chosen=args.only.split(',') if args.only else config['preflight_scenes'];assert set(chosen)<=set(ids)
        preview(config,out,chosen);return
    if args.phase=='worker':
        chosen=args.only.split(',');assert set(chosen)<=set(ids);worker(config,out,chosen);return
    if args.phase=='controls':controls(config,out);return
    if args.phase=='evaluate':evaluate(config,out,ids);return
    assert 1<=args.workers<=8
    logs=ROOT/config['logs'];logs.mkdir(parents=True,exist_ok=True)
    if stage=='full':
        pre=ROOT/config['preflight_output'];accept=json.loads((pre/'acceptance.json').read_text())
        assert accept['status']=='PREFLIGHT_PASSED' and accept['protocol']==config
        for sid in config['preflight_scenes']:
            for category in ['clips','timing']:
                src=pre/category/sid;dst=out/category/sid;dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():assert dst.resolve()==src.resolve()
                else:dst.symlink_to(src,target_is_directory=True)
    pending=[s for s in ids if not (out/'timing'/s/'complete.json').exists()]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',TF_ENABLE_ONEDNN_OPTS='0',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='4')
    children=[];jobs=[];stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for i in range(args.workers):
        chosen=pending[i::args.workers]
        if not chosen:continue
        log=logs/f'{stage}-{stamp}-worker{i}.log'
        command=[str(NATIVE),'-u',str(Path(__file__).resolve()),'--config',args.config,'--phase','worker','--stage',stage,'--only',','.join(chosen)]
        with log.open('x') as f:child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
        children.append(child);jobs.append(dict(pid=child.pid,scenes=chosen,log=str(log.relative_to(ROOT))))
    write(out/f'workers-{stamp}.json',dict(supervisor=os.getpid(),workers=jobs))
    codes=[c.wait() for c in children];assert all(c==0 for c in codes),('Worker failed; preserve outputs',codes)
    check(config,out,ids,stage)
    if stage=='preflight':
        with (logs/'control-check.log').open('a') as f:
            subprocess.run([str(NATIVE),'-u',str(Path(__file__).resolve()),'--config',args.config,'--phase','controls'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('DEEPLENS_PREFLIGHT_NUMERICAL_COMPLETE_VISUAL_PENDING',flush=True)
    else:
        gpu_env=os.environ.copy();gpu_env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        subprocess.run([str(METRIC_PY),'-u',str(Path(__file__).resolve()),'--config',args.config,'--phase','evaluate','--stage','full'],env=gpu_env,check=True)


if __name__=='__main__':main()
