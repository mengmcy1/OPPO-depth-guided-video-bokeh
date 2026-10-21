"""Approved revision1 B36 paired inference. Original model, fresh processes.

Input preparation reconstructs the exact phase1 proposed float CoC; color PNGs
are verified against it, never used as numeric model input. No selection rule
or source-frame/padding policy is changed. Large arrays stay under outputs.
"""
import argparse
from datetime import datetime, timezone
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

from check_atb_mpi_invariance import author_imports, load_model, digest, rng_state, tensor_record, compare, KEYS
from prepare_b36_rule_b import original_geometry, color, native_box, display_box, match

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/any2bokeh-B36-rule-b-inference-20261021.json'


def read(p):return json.loads(Path(p).read_text())


def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),f'Preserve existing {p}'
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prepare(c,out):
    import torchvision.transforms.functional as F
    approval=read(ROOT/c['approval_record']);assert approval['user_selection_confirmed']
    selection=read(ROOT/c['selection_config']);result=read(ROOT/c['selection_record'])
    assert selection['candidate_rule']['revision']==c['selection_revision']==1
    assert result['config']['candidate_rule']==selection['candidate_rule']
    assert not out.exists();out.mkdir(parents=True)
    old=ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-MPI-invariance-20261021'
    prior=read(old/'prepared.json');old_inputs=torch.load(old/'prepared-inputs.pt',weights_only=True,map_location='cpu')['base']
    _,_,_,_,Dataset,collate=author_imports()
    csvpath=ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-causal-diagnosis-20261015/inputs/native/input.csv'
    base=collate([Dataset(str(csvpath),sample_size=(576,1024))[0]])
    base={k:base[k] for k in KEYS}
    assert all(digest(base[k])==digest(old_inputs[k])==prior['original'][k]['sha256'] for k in KEYS)
    del old_inputs
    groups=prior['audit']['input_groups'];candidate=base['coc'].clone();rows=[]
    source=ROOT/selection['output']/'B36'
    for t,row in enumerate(result['rows']):
        assert row['frame_id']==t
        q,union,focus=original_geometry(selection,t)
        chosen=np.asarray(Image.open(source/'native/Selected_mask'/f'{t:05d}.png'))!=0
        assert int(chosen.sum())==row['selected_area'] and not chosen[~union].any()
        assert bool(chosen.any())==(t<=20)
        qt=torch.from_numpy(q.astype(np.float32));modified=qt.clone()
        if chosen.any():modified[torch.from_numpy(chosen)]=row['reference_q']
        coc=(modified-focus*255).abs()/255
        value=F.normalize(F.resize(coc.repeat(3,1,1),(576,1024)),[.5]*3,[.5]*3)
        shown=color(((value[0]+1)/2).numpy()*16,0,16)
        assert np.array_equal(shown,np.asarray(Image.open(source/'originals/Proposed_CoC'/f'{t:05d}.png')))
        slots=[]
        for g,ids in enumerate(groups):
            for j,f in enumerate(ids):
                if f==t:
                    before=base['coc'][g,j];delta=value-before
                    assert int(torch.count_nonzero(delta))==row['condition_changed_elements']
                    assert float(delta.abs().max())==row['condition_max_abs_change']
                    assert float(delta.abs().mean())==row['condition_mean_abs_change']
                    candidate[g,j]=value;slots.append([g,j])
        if t>=21:assert all(torch.equal(candidate[g,j],base['coc'][g,j]) for g,j in slots)
        rows.append(dict(frame=t,focus=focus,K=16,selected_native_pixels=int(chosen.sum()),
            slots=slots,condition_changed=t<=20,CoC_sha256=digest(value),
            phase1_numeric_and_rendered_CoC_equal=True,skip_reason=row['missing_or_skip_reason']))
    valid=torch.zeros((6,8),dtype=torch.bool)
    for g,ids in enumerate(groups):valid[g,:len(ids)]=True
    assert int(valid.sum())==45 and torch.equal(candidate[~valid],base['coc'][~valid])
    torch.save(dict(base=base,rule_b_coc=candidate,valid=valid),out/'inputs.pt')
    save(out/'prepared.json',dict(config=c,approval=approval,selection_rule=selection['candidate_rule'],
        created_UTC=datetime.now(timezone.utc).isoformat(),groups=groups,padding_slots=[[5,5],[5,6],[5,7]],
        original={k:tensor_record(v) for k,v in base.items()},rule_b_coc=tensor_record(candidate),
        input_rows=rows,legacy_CPU_inputs_bitwise_equal=True,
        note='Fresh local A and A-repeat will be run. Old outputs are not used as paired reference. No phase1 color image is decoded into a condition.'))
    print('PREPARED',out,flush=True)


def worker(c,out,condition):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    dest=out/condition;assert not dest.exists();dest.mkdir()
    pipe,p=load_model();inputs=torch.load(out/'inputs.pt',weights_only=True,map_location='cpu')
    base=inputs['base'];batch=dict(base)
    if condition=='rule_b':batch['coc']=inputs['rule_b_coc']
    for k in ['pixel_values','k','mpi_mask','weight']:assert digest(batch[k])==digest(base[k])
    assert torch.equal(batch['coc'][~inputs['valid']],base['coc'][~inputs['valid']])
    prepared=read(out/'prepared.json');legacy=ROOT/'outputs/davis-baselines-20260929/any2bokeh'
    order=read(legacy/'protocol.json')['scene_order'];i=order.index(c['scene'])
    rng_path=legacy/'audit'/order[i-1]/'rng-after.pt';state=torch.load(rng_path,weights_only=True)
    assert state['next_scene_index']==i==33
    assert not hasattr(pipe.vae.encoder,'residual_down_blocks')
    cache=None;torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'],device=0)
    events=dict(posterior=[],units=[]);current={}
    from models.vae import DiagonalGaussianDistribution
    original_encode=pipe.vae.encode;original_sample=DiagonalGaussianDistribution.sample
    def encode_observer(*args,**kwargs):
        current['branch']='RGB' if kwargs.get('cache_flag',False) else 'CoC'
        return original_encode(*args,**kwargs)
    def sample_observer(distribution,*args,**kwargs):
        before=rng_state();value=original_sample(distribution,*args,**kwargs)
        events['posterior'].append(dict(unit=current['unit'],branch=current['branch'],shape=list(value.shape),
            rng_before=before,rng_after=rng_state(),generator='global/default' if not args and kwargs.get('generator') is None else 'explicit',
            sample_sha256=digest(value)))
        return value
    pipe.vae.encode=encode_observer;DiagonalGaussianDistribution.sample=sample_observer
    runtime=dict(condition=condition,pid=os.getpid(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),author_commit=p['author_commit'],entry_RNG=rng_state(),
        RNG_source=str(rng_path.relative_to(ROOT)),initial_encoder_cache_absent=True,initial_latent_cache=None,
        imports={name:inspect.getfile(type(obj)) for name,obj in [('pipeline',pipe),('unet',pipe.unet),('vae',pipe.vae)]},
        input_sha256={k:digest(v) for k,v in batch.items()},
        allocator_environment={k:v for k,v in os.environ.items() if k.startswith(('PYTORCH_ALLOC','PYTORCH_CUDA_ALLOC'))})
    save(dest/'runtime.json',runtime)
    output=np.lib.format.open_memmap(dest/'float.npy',mode='w+',dtype=np.float32,shape=(25,576,1024,3))
    (dest/'png').mkdir();offset=0;start=time.monotonic()
    for unit,g in enumerate([0,2,4]):
        current['unit']=unit
        tensors={k:batch[k][g:g+2].to('cuda',dtype=torch.float32 if k in ['pixel_values','coc'] else torch.float16) for k in KEYS}
        generator=torch.Generator(device='cpu').manual_seed(0)
        event=dict(unit=unit,groups=prepared['groups'][g:g+2],rng_before=rng_state(),
            CPU_generator_before=digest(generator.get_state()),latent_cache_in=None if cache is None else tensor_record(cache),
            input_sha256={k:digest(v) for k,v in tensors.items()})
        with torch.inference_mode():
            result,cache=pipe(tensors['coc'],tensors['pixel_values'],input_k=tensors['k'],mpi_mask=tensors['mpi_mask'],
                weight=tensors['weight'],latent_cache=cache,is_last_batch=g==4,overlap_frame=4,height=576,width=1024,
                num_frames=8,decode_chunk_size=8,motion_bucket_id=127,fps=7,noise_aug_strength=0.,num_inference_steps=1,
                generator=generator,batch_size=2,output_type='pt')
        assert tuple(result[0].shape)==((9 if g==4 else 8),3,576,1024) and torch.isfinite(result[0]).all()
        arr=result[0].detach().cpu().permute(0,2,3,1).numpy();output[offset:offset+len(arr)]=arr
        for j,image in enumerate(arr):Image.fromarray(np.uint8(np.round(np.clip(image,0,1)*255))).save(dest/'png'/f'{offset+j:05d}.png')
        offset+=len(arr);event.update(output_frames=len(arr),rng_after=rng_state(),CPU_generator_after=digest(generator.get_state()),latent_cache_out=tensor_record(cache))
        events['units'].append(event);print('CHUNK',condition,unit,offset,flush=True)
        del result,tensors,arr
    assert offset==25;output.flush();pipe.vae.encode=original_encode;DiagonalGaussianDistribution.sample=original_sample
    save(dest/'events.json',events)
    save(dest/'complete.json',dict(condition=condition,shape=list(output.shape),finite=bool(np.isfinite(output).all()),
        frames=25,pngs=len(list((dest/'png').glob('*.png'))),end_RNG=rng_state(),seconds_with_saving=time.monotonic()-start,
        performance_measurement=False))
    print('DONE',condition,flush=True)


def run(c,out):
    assert read(out/'prepared.json')['config']==c
    for condition in c['conditions']:
        if (out/condition/'complete.json').exists():
            print('REUSE_COMPLETE',condition,flush=True)
        else:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy; preserve other process'
            with (out/f'{condition}.log').open('x') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--condition',condition],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if condition=='native_repeat':
            check=compare(out,condition)
            if (out/'repeat-check.json').exists():assert read(out/'repeat-check.json')==check
            else:save(out/'repeat-check.json',check)
            print('REPEAT', {k:v for k,v in check.items() if k!='per_frame'},flush=True)
    print('ALL_THREE_INFERENCE_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','worker','run'])
    parser.add_argument('--condition',choices=['native','native_repeat','rule_b']);args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);c=read(CFG);out=ROOT/c['output']
    if args.stage=='worker':assert args.condition;worker(c,out,args.condition)
    else:globals()[args.stage](c,out)
