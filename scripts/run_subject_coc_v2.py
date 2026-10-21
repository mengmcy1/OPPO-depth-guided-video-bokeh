"""Fixed GT-union subject CoC constraint, six exploration sources, tau1.

Only CoC is transplanted. The public model, MPI, focus/K, precision, overlap,
sampling and cache lifecycle remain native. Config must be committed before GPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
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

from check_atb_mpi_invariance import author_imports,load_model,digest,rng_state,tensor_record,KEYS
from prepare_b36_rule_b import ROOT,read,save,now
from run_b36_rule_b_r2 import bit_equal
from analyze_b36_rule_b import bbox,rectangle,link
from stat_subject_coc import values,masks_for,write_csv

CFG=ROOT/'configs/any2bokeh-subject-coc-clamp-v2-20261021.json'


def csv_rows(path,rows):
    from analyze_b36_rule_b import csv_write
    csv_write(path,rows)


def geometry(scene,t):
    src=ROOT/'data/processed/davis2016-v2/eval'/scene
    meta=read(ROOT/'data/processed/davis2016-v2/sequences'/scene/'geometry.json')['frames'][t]
    q=np.asarray(Image.open(src/'disparity'/meta['file']))
    label=np.asarray(Image.open(ROOT/'data/raw/DAVIS-2017/DAVIS/Annotations/Full-Resolution'/scene/f'{t:05d}.png'))
    assert label.shape==q.shape and float(Path(meta['file']).stem.split('_zf_')[1])==meta['focus']
    assert np.array_equal(label!=0,np.asarray(Image.open(src/'mask'/f'{t:05d}.png'))!=0)
    return q,label,meta['focus']


def condition(ds,q,focus):
    x=(torch.from_numpy(q.astype(np.float32))-focus*255).abs().div(255).repeat(3,1,1)
    for tr in ds.pixel_transforms.transforms[:-1]:x=tr(x)
    before=x.clone();norm=ds.pixel_transforms.transforms[-1](x)
    return norm,before[0].numpy()*16


def prepare(c,out):
    assert c['tau']==1 and c['K']==16 and len(c['samples'])==6
    _,_,_,_,Dataset,collate=author_imports()
    split=read(ROOT/c['split_record'])['rows'];allowed={r['scene_id'] for r in split if r['dataset']=='DAVIS' and r['split']=='exploration'}
    legacy=ROOT/'outputs/davis-baselines-20260929/any2bokeh';order=read(legacy/'protocol.json')['scene_order']
    for sample in c['samples']:
        scene=sample['scene_id'];assert scene in allowed
        dest=out/'scenes'/scene
        if (dest/'prepared.json').exists():assert read(dest/'prepared.json')['config']==c;print('REUSE_INPUTS',scene,flush=True);continue
        dest.mkdir(parents=True,exist_ok=True)
        ds=Dataset(str(legacy/'audit'/scene/'input.csv'),sample_size=(576,1024));base=collate([ds[0]]);base={k:base[k] for k in KEYS}
        audit=read(legacy/'audit'/scene/'input.json');groups=audit['input_groups'];assert groups==[list(range(s,min(s+8,25))) for s in [0,4,8,12,16,20]]
        valid=torch.zeros((6,8),dtype=torch.bool)
        for g,ids in enumerate(groups):valid[g,:len(ids)]=True
        candidate=base['coc'].clone();all_ids=sorted({int(v) for t in range(25) for v in np.unique(geometry(scene,t)[1]) if v})
        input_rows=[];region_frames=[];frame_checks=[]
        b36=None
        if scene=='mallard-water':
            prior=ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-rule-b-inference-20261021'
            old=torch.load(prior/'inputs.pt',weights_only=True,map_location='cpu')
            assert all(bit_equal(base[k],old['base'][k]) for k in KEYS);del old
            b36=read(ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-rule-b-R2-20261021/regions.json')
        for t in range(25):
            q,label,focus=geometry(scene,t);union=label!=0;d=q.astype(np.float64)/255
            assert q.shape==(1080,1920), 'The frozen six-source V2 config specifies this native grid'
            lower,upper=focus-1/16,focus+1/16
            limited=union&((d<lower)|(d>upper));changed=d.copy();changed[union]=np.clip(d[union],lower,upper)
            assert np.array_equal(changed[~union],d[~union])
            before,before_display=condition(ds,q,focus);after,after_display=condition(ds,changed*255,focus)
            positions=[[g,j] for g,ids in enumerate(groups) for j,source in enumerate(ids) if source==t]
            for g,j in positions:
                assert bit_equal(before,base['coc'][g,j]);candidate[g,j]=after
            small=lambda a:cv2.resize(a.astype(np.uint8),(1024,576),interpolation=cv2.INTER_NEAREST)>0
            union_d=small(union);limited_d=small(limited)&union_d
            exterior=cv2.distanceTransform((~union_d).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            native_before=16*np.abs(d-focus);native_after=16*np.abs(changed-focus)
            assert (native_after[union]<=1+1e-12).all()
            spread=int(((after[0].numpy()!=before[0].numpy())&~union_d).sum())
            frame_checks.append(dict(frame_id=t,limited_native_pixels=int(limited.sum()),
                resized_union_outside_changed_pixels=spread,all_real_slots=positions,
                native_union_after_max=float(native_after[union].max()),
                actual_union_after_max=float(after_display[union_d].max()),
                padding_unchanged=True))
            folder=dest/'regions'/f'{t:05d}';folder.mkdir(parents=True)
            regions=[]
            for label_id in ['union']+list(map(str,all_ids)):
                m=union if label_id=='union' else label==int(label_id);md=small(m)
                distance=exterior if label_id=='union' else cv2.distanceTransform((~md).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                sets=dict(limited=limited_d&md,remainder=md&~limited_d,
                    exterior_ring_2_6=(~union_d)&(distance>=2)&(distance<=6),far_background=(~union_d)&(exterior>32))
                for region,mask in sets.items():
                    path=folder/(label_id+'-'+region+'.png');Image.fromarray(np.uint8(mask)*255).save(path)
                    regions.append(dict(label_id=label_id,region=region,path=str(path.relative_to(dest)),pixels=int(mask.sum()),display_bbox=bbox(mask)))
                partitions=masks_for(label,label_id)
                for grid,fields in partitions.items():
                    prior_a,new_a=(native_before,native_after) if grid=='native' else (before_display,after_display)
                    for region,mask in fields.items():
                        row=dict(scene_id=scene,sample_id=sample['sample_id'],frame_id=t,label_id=label_id,grid=grid,region=region,
                            native_width=q.shape[1],native_height=q.shape[0],focus=focus,K=16,tau=1,
                            limited_native_pixels=int((limited&m).sum()),subject_native_pixels=int(m.sum()),
                            limited_fraction_subject=None if not m.any() else float((limited&m).sum()/m.sum()),
                            resized_union_outside_changed_pixels=spread,
                            **{'before_'+k:v for k,v in values(prior_a,mask).items()},**{'after_'+k:v for k,v in values(new_a,mask).items()})
                        input_rows.append(row)
            if b36 is not None:
                r=next(x for x in b36['frames'][t]['regions'] if x['region']=='target_head')
                source=ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-rule-b-R2-20261021'/r['path']
                path=folder/'union-B36_head_target.png';link(path,source)
                regions.append(dict(label_id='union',region='B36_head_target',path=str(path.relative_to(dest)),pixels=r['pixels'],display_bbox=r['display_bbox']))
            region_frames.append(dict(frame_id=t,subject_bbox=bbox(union_d),limited_bbox=bbox(limited_d),regions=regions))
        assert bit_equal(candidate[~valid],base['coc'][~valid])
        torch.save(dict(base=base,candidate_coc=candidate,valid=valid),dest/'inputs.pt')
        index=order.index(scene);assert index>0
        save(dest/'prepared.json',dict(config=c,scene_id=scene,sample=sample,groups=groups,scene_index=index,
            original={k:tensor_record(v) for k,v in base.items()},candidate_coc=tensor_record(candidate),
            RNG_source=str((legacy/'audit'/order[index-1]/'rng-after.pt').relative_to(ROOT)),
            label_ids=all_ids,instance_type='multi-instance' if len(all_ids)>1 else 'single-instance',
            mask_permission='DAVIS GT ID!=0 union expanded from scalar focus to per-pixel CoC constraint; no generator mask or MPI change.',
            input_rows=input_rows,frame_checks=frame_checks,all25_rule_applied=True,no_frame_skips=True,padding_unchanged=True))
        save(dest/'regions.json',dict(frozen_before_outputs=True,policy=c['region_policy'],frames=region_frames))
        print('PREPARED',scene,'IDs',all_ids,flush=True)
    csv_rows(ROOT/(c['record_prefix']+'-inputs.csv'),[r for s in c['samples'] for r in read(out/'scenes'/s['scene_id']/'prepared.json')['input_rows']])


def worker(c,out,scene,variant):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    dest=out/'scenes'/scene;run=dest/variant;assert not run.exists();run.mkdir()
    pipe,p=load_model();prepared=read(dest/'prepared.json');loaded=torch.load(dest/'inputs.pt',weights_only=True,map_location='cpu')
    batch=dict(loaded['base'])
    if variant=='V2':batch['coc']=loaded['candidate_coc']
    assert all(digest(batch[k])==prepared['original'][k]['sha256'] for k in ['pixel_values','k','mpi_mask','weight'])
    assert bit_equal(batch['coc'][~loaded['valid']],loaded['base']['coc'][~loaded['valid']])
    state=torch.load(ROOT/prepared['RNG_source'],weights_only=True);assert state['next_scene_index']==prepared['scene_index']
    assert not hasattr(pipe.vae.encoder,'residual_down_blocks')
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'],device=0);cache=None
    events=dict(posterior=[],units=[]);current={}
    from models.vae import DiagonalGaussianDistribution
    encode=pipe.vae.encode;sample=DiagonalGaussianDistribution.sample
    def encode_observer(*args,**kwargs):current['branch']='RGB' if kwargs.get('cache_flag',False) else 'CoC';return encode(*args,**kwargs)
    def sample_observer(distribution,*args,**kwargs):
        start=rng_state();value=sample(distribution,*args,**kwargs)
        events['posterior'].append(dict(unit=current['unit'],branch=current['branch'],shape=list(value.shape),rng_before=start,rng_after=rng_state(),sha256=digest(value)))
        return value
    pipe.vae.encode=encode_observer;DiagonalGaussianDistribution.sample=sample_observer
    save(run/'runtime.json',dict(scene_id=scene,variant=variant,pid=os.getpid(),author_commit=p['author_commit'],
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
        input_sha256={k:digest(v) for k,v in batch.items()},entry_RNG=rng_state(),RNG_source=prepared['RNG_source'],initial_latent_cache=None,
        imports={k:inspect.getfile(type(v)) for k,v in [('pipeline',pipe),('vae',pipe.vae),('unet',pipe.unet)]}))
    arr=np.lib.format.open_memmap(run/'float.npy',mode='w+',dtype=np.float32,shape=(25,576,1024,3));(run/'png').mkdir();offset=0
    for unit,g in enumerate([0,2,4]):
        current['unit']=unit;keys={k:batch[k][g:g+2].to('cuda',dtype=torch.float32 if k in ['pixel_values','coc'] else torch.float16) for k in KEYS}
        gen=torch.Generator(device='cpu').manual_seed(0)
        event=dict(unit=unit,groups=prepared['groups'][g:g+2],rng_before=rng_state(),CPU_generator_before=digest(gen.get_state()),input_sha256={k:digest(v) for k,v in keys.items()})
        with torch.inference_mode():
            value,cache=pipe(keys['coc'],keys['pixel_values'],input_k=keys['k'],mpi_mask=keys['mpi_mask'],weight=keys['weight'],latent_cache=cache,
                is_last_batch=g==4,overlap_frame=4,height=576,width=1024,num_frames=8,decode_chunk_size=8,motion_bucket_id=127,fps=7,
                noise_aug_strength=0.,num_inference_steps=1,generator=gen,batch_size=2,output_type='pt')
        assert tuple(value[0].shape)==((9 if g==4 else 8),3,576,1024) and torch.isfinite(value[0]).all()
        x=value[0].detach().cpu().permute(0,2,3,1).numpy();arr[offset:offset+len(x)]=x
        for j,im in enumerate(x):Image.fromarray(np.uint8(np.round(np.clip(im,0,1)*255))).save(run/'png'/f'{offset+j:05d}.png')
        offset+=len(x);event.update(output_frames=len(x),rng_after=rng_state(),CPU_generator_after=digest(gen.get_state()));events['units'].append(event)
        print('CHUNK',scene,variant,unit,offset,flush=True);del value,keys,x
    assert offset==25;arr.flush();save(run/'events.json',events)
    save(run/'complete.json',dict(scene_id=scene,variant=variant,frames=25,shape=list(arr.shape),finite=True,end_RNG=rng_state()))


def launch_one(c,out,scene,variant):
    run=out/'scenes'/scene/variant
    if (run/'complete.json').exists():print('REUSE_COMPLETED',scene,variant,flush=True);return
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy; do not stop other tasks'
    with (out/f'{scene}-{variant}.log').open('x') as f:subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--scene',scene,'--variant',variant],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)


def run(c,out):
    committed=subprocess.check_output(['git','show','HEAD:'+str(CFG.relative_to(ROOT))],cwd=ROOT)
    assert committed==CFG.read_bytes(),'Freeze and commit exact config before inference'
    if (out/'inference-complete.json').exists():print('REUSE_ALL_SIX_COMPLETED',flush=True);return
    if not (out/'freeze.json').exists():save(out/'freeze.json',dict(config_sha256=hashlib.sha256(committed).hexdigest(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),UTC=now()))
    launch_one(c,out,'rhino','A-check')
    probe=out/'scenes/rhino/A-check/png';frozen=ROOT/'outputs/davis-baselines-20260929/any2bokeh/predictions/rhino'
    checks=[]
    for t in range(25):
        a=np.asarray(Image.open(frozen/f'{t:05d}.png'));b=np.asarray(Image.open(probe/f'{t:05d}.png'));d=np.abs(a.astype(np.int16)-b.astype(np.int16))
        checks.append(dict(frame_id=t,RGB8_different_bytes=int(np.count_nonzero(a!=b)),RGB8_max_abs_diff=int(d.max()),RGB8_MAE=float(d.mean())))
    match=all(r['RGB8_different_bytes']==0 for r in checks)
    reuse=dict(B18_RGB8_bitwise_equal=match,per_frame=checks,
        policy='Reuse frozen A for other scenes' if match else 'All six use current-environment A: B18 probe reused, B36 already-current paired A reused as explicitly required; other four A rerun.',
        old_frozen_float_available=False)
    if (out/'A-reproduction.json').exists():assert read(out/'A-reproduction.json')==reuse
    else:save(out/'A-reproduction.json',reuse)
    print('A_REPRODUCTION',match,'different_bytes',sum(r['RGB8_different_bytes'] for r in checks),flush=True)
    for sample in c['samples']:
        scene=sample['scene_id'];dest=out/'scenes'/scene
        if not (dest/'A').exists():
            if scene=='mallard-water':link(dest/'A',ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-rule-b-inference-20261021/native')
            elif scene=='rhino':link(dest/'A',dest/'A-check')
            elif match:
                (dest/'A').mkdir();link(dest/'A/png',ROOT/'outputs/davis-baselines-20260929/any2bokeh/predictions'/scene)
                save(dest/'A/source.json',dict(source='frozen DAVIS original PNG',RGB8_only=True))
            else:launch_one(c,out,scene,'A')
        launch_one(c,out,scene,'V2')
    save(out/'inference-complete.json',dict(UTC=now(),scenes=[s['scene_id'] for s in c['samples']],V2_frames=150,
        new_A_runs=1 if match else 5,B36_A_reused=True,A_policy=reuse['policy']))
    print('ALL_SIX_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run','worker']);parser.add_argument('--scene');parser.add_argument('--variant',choices=['A-check','A','V2']);a=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);c=read(CFG);out=ROOT/c['output']
    if a.stage=='worker':worker(c,out,a.scene,a.variant)
    else:globals()[a.stage](c,out)
