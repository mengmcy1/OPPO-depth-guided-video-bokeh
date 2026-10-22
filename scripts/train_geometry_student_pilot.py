"""Fixed three-arm geometry-only pilot. Freeze config before GPU training."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.geometry_student import GeometryResidualUNet,predict
CFG=ROOT/'configs/ours-student-pilot-v1-20261023.json';C=json.loads(CFG.read_text());OUT=ROOT/C['output']


def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def model_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def schedule():
    rng=np.random.default_rng(C['schedule_seed']);n=C['steps']*C['batch_size']
    indices=np.concatenate([rng.permutation(100) for _ in range((n+99)//100)])[:n]
    top=rng.integers(0,436-C['crop_hw'][0]+1,size=n)
    left=rng.integers(0,1024-C['crop_hw'][1]+1,size=n)
    flip=rng.integers(0,2,size=n)
    branch_rng=np.random.default_rng(C['branch_seed']);branches=[]
    for _ in range(C['steps']):
        b=np.array([0,0,1,1]);branch_rng.shuffle(b);branches.extend(b)
    plan=np.stack([indices,top,left,flip,np.asarray(branches)],1).astype(np.int64)
    return plan,hashlib.sha256(plan.tobytes()).hexdigest()


def load_cache():
    manifest=json.loads((ROOT/C['data_manifest']).read_text());data_config=json.loads((ROOT/C['data_config']).read_text())
    assert manifest['config']==data_config
    entries=manifest['frames'];arrays={k:[] for k in ['rgb','natural','perturbed','reference','valid','near','footprint','boundary']}
    scenes=[];lookup={};i=0
    for row in manifest['summary']:
        name=row['sequence'];rows=[r for r in entries if r['sequence']==name]
        assert len(rows)==25 and [r['array_index'] for r in rows]==list(range(25))
        with np.load(ROOT/rows[0]['vda_npz']) as z:arrays['natural'].append(z['disparity_normalized'])
        with np.load(ROOT/rows[0]['pair_npz']) as z:
            for key,source in [('perturbed','perturbed'),('reference','reference'),('valid','supervision_domain'),('near','near_reference'),('footprint','perturbation_mask'),('boundary','boundary_diagnostic')]:arrays[key].append(z[source])
        rgb=np.stack([np.asarray(Image.open(ROOT/r['rgb']).convert('RGB')) for r in rows]).transpose(0,3,1,2).copy()
        arrays['rgb'].append(rgb)
        scenes.append(dict(sequence=name,split=row['split'],indices=list(range(i,i+25))))
        for r in rows:lookup[i]=r;i+=1
    train=[j for s in scenes if s['split']=='pilot_train' for j in s['indices']]
    dev=[j for s in scenes if s['split']=='pilot_dev' for j in s['indices']]
    assert train==list(range(100)) and dev==list(range(100,150))
    assert {lookup[j]['source_id'] for j in train}.isdisjoint(lookup[j]['source_id'] for j in dev)
    cache={k:torch.from_numpy(np.concatenate(v)).to('cuda') for k,v in arrays.items()}
    assert all(torch.isfinite(v).all() for k,v in cache.items() if v.is_floating_point())
    return cache,scenes,lookup,data_config


def crop_batch(cache,plan,variant):
    h,w=C['crop_hw'];xs=[];ys=[];valid=[];keep=[];changed=0;branches=[];footprints=0
    for idx,top,left,flip,joint_branch in plan:
        branch=0 if variant=='natural' else 1 if variant=='perturbed' else joint_branch
        branches.append(int(branch));sl=(slice(top,top+h),slice(left,left+w))
        rgb=cache['rgb'][idx,:,sl[0],sl[1]].float()/255
        d=cache['perturbed' if branch else 'natural'][idx][sl]
        mask=cache['footprint'][idx][sl]
        x=torch.cat([rgb,d[None]],0);y=cache['reference'][idx][sl][None]
        v=cache['valid'][idx][sl][None];k=cache['near'][idx][sl][None]
        if branch:k=k&~mask[None]
        if flip:x=x.flip(-1);y=y.flip(-1);v=v.flip(-1);k=k.flip(-1)
        xs.append(x);ys.append(y);valid.append(v);keep.append(k)
        if branch:
            changed+=int(torch.count_nonzero(d!=cache['natural'][idx][sl]))
            footprints+=int(mask.sum())
    return torch.stack(xs),torch.stack(ys),torch.stack(valid),torch.stack(keep),branches,changed,footprints


def objective(out,x,target,valid,keep,eps):
    geo=((out-target).abs()*valid).sum()/valid.sum().clamp_min(1)
    penalty=F.relu((out-x[:,3:4]).abs()-eps)
    preservation=(penalty*keep).sum()/keep.sum().clamp_min(1)
    return geo+C['keep_weight']*preservation,geo,preservation


@torch.inference_mode()
def evaluate(model,cache,scenes,splits,output=None):
    model.eval();rows=[]
    for scene in scenes:
        if scene['split'] not in splits:continue
        for branch in ['natural','perturbed']:
            predictions=[]
            for frame,idx in enumerate(scene['indices']):
                d=cache[branch][idx];target=cache['reference'][idx]
                x=torch.cat([cache['rgb'][idx].float()/255,d[None]],0)[None]
                with torch.autocast('cuda',dtype=torch.bfloat16):out=predict(model,x)[0,0]
                assert torch.isfinite(out).all();clipped=out.clamp(0,1)
                pred=clipped.cpu().numpy();raw=out.cpu().numpy();ref=target.cpu().numpy();inp=d.cpu().numpy()
                domain=cache['valid'][idx].cpu().numpy();edge=cache['boundary'][idx].cpu().numpy();near=cache['near'][idx].cpu().numpy();foot=cache['footprint'][idx].cpu().numpy()
                masks=dict(valid=domain,boundary=domain&edge,interior=domain&~edge,near_reference=domain&near,
                           near_untouched=domain&near&~foot,perturbation_footprint=domain&foot,unsupported=~domain)
                for region,m in masks.items():
                    n=int(m.sum());supported=region!='unsupported'
                    rows.append(dict(scene=scene['sequence'],split=scene['split'],frame=frame,branch=branch,region=region,pixels=n,
                                     baseline_mae=float(np.abs(inp-ref)[m].mean(dtype=np.float64)) if n and supported else None,
                                     mae=float(np.abs(pred-ref)[m].mean(dtype=np.float64)) if n and supported else None,
                                     preclamp_mae=float(np.abs(raw-ref)[m].mean(dtype=np.float64)) if n and supported else None,
                                     mse=float(((pred-ref)**2)[m].mean(dtype=np.float64)) if n and supported else None,
                                     mean_change=float(np.abs(pred-inp)[m].mean(dtype=np.float64)) if n else None,
                                     outside_0_1_fraction=float(((raw<0)|(raw>1))[m].mean()) if n else None))
                if output is not None and scene['split']=='pilot_dev':predictions.append(pred)
            if predictions:
                dest=output/'predictions'/scene['sequence'];dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/(branch+'.npz'),prediction=np.stack(predictions))
    aggregated=[]
    for scene in scenes:
        if scene['split'] not in splits:continue
        for branch in ['natural','perturbed']:
            for region in ['valid','boundary','interior','near_reference','near_untouched','perturbation_footprint','unsupported']:
                selected=[r for r in rows if r['scene']==scene['sequence'] and r['branch']==branch and r['region']==region and r['pixels']]
                n=sum(r['pixels'] for r in selected)
                rec=dict(scene=scene['sequence'],split=scene['split'],branch=branch,region=region,pixels=n)
                for field in ['baseline_mae','mae','preclamp_mae','mse','mean_change','outside_0_1_fraction']:
                    rec[field]=sum(r[field]*r['pixels'] for r in selected)/n if n and all(r[field] is not None for r in selected) else None
                aggregated.append(rec)
    model.train();return dict(per_scene=aggregated,per_frame=rows)


def quick_summary(result,split='pilot_dev',branch='natural',region='valid'):
    rs=[r for r in result['per_scene'] if r['split']==split and r['branch']==branch and r['region']==region]
    return {k:float(np.mean([r[k] for r in rs if r[k] is not None])) for k in ['baseline_mae','mae','preclamp_mae','mean_change']}


def smoke():
    torch.set_num_threads(4);torch.manual_seed(C['model_seed'])
    model=GeometryResidualUNet();x=torch.rand(2,4,32,32);y=torch.rand(2,1,32,32);v=torch.ones_like(y,dtype=torch.bool);k=torch.zeros_like(v)
    assert sum(p.numel() for p in model.parameters())==471841
    assert torch.equal(predict(model,x),x[:,3:4])
    optimizer=torch.optim.AdamW(model.parameters(),lr=C['learning_rate'])
    for _ in range(2):
        optimizer.zero_grad();out=predict(model,x);loss,_,_=objective(out,x,y,v,k,.028637895);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model.e1[0].weight.grad.abs().sum()>0,'Encoder cannot learn after residual head updates'
    plan,h=schedule();assert plan.shape==(8000,5) and np.array_equal(plan,schedule()[0])
    assert np.array_equal(np.bincount(plan[:,0]),np.full(100,80))
    assert all(plan[i:i+4,4].sum()==2 for i in range(0,len(plan),4))
    save_json(ROOT/(C['record_prefix']+'-preflight.json'),dict(status='CPU_SMOKE_PASSED',parameters=471841,zero_head_is_identity=True,
        second_step_encoder_gradient_nonzero=True,schedule_sha256=h,per_frame_exposures=80,total_exposures=8000,joint_counts=[4000,4000],
        scratch_CPU_updates=2,formal_training_steps=0))
    print('CPU_SMOKE_PASSED',h,flush=True)


def train(variant):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert subprocess.check_output(['git','show','HEAD:'+str(CFG.relative_to(ROOT))],cwd=ROOT)==CFG.read_bytes(),'Commit frozen config first'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
    dest=OUT/variant;assert not dest.exists();dest.mkdir(parents=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.manual_seed(C['model_seed'])
    start=time.monotonic();cache,scenes,lookup,dc=load_cache();load_seconds=time.monotonic()-start
    model=GeometryResidualUNet().cuda();init=model_hash(model);plan,plan_hash=schedule()
    optimizer=torch.optim.AdamW(model.parameters(),lr=C['learning_rate'],weight_decay=C['weight_decay'],betas=tuple(C['betas']))
    save_json(dest/'runtime.json',dict(config=C,variant=variant,init_sha256=init,schedule_sha256=plan_hash,
        torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(0),data_load_seconds=load_seconds,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),started_utc=datetime.now(timezone.utc).isoformat()))
    baseline=evaluate(model,cache,scenes,['pilot_train','pilot_dev'])
    for r in baseline['per_scene']:
        if r['mae'] is not None:assert abs(r['mae']-r['baseline_mae'])<1e-10
    save_json(dest/'eval-00000.json',baseline)
    print('START',variant,'dev',quick_summary(baseline),flush=True)
    torch.cuda.reset_peak_memory_stats();begin=time.monotonic();eval_seconds=0.;branch_counts=[0,0];changed_total=0;footprint_total=0;curve=[]
    with (dest/'train.jsonl').open('x') as log:
        for step in range(1,C['steps']+1):
            x,y,v,k,branches,changed,footprints=crop_batch(cache,plan[(step-1)*4:step*4],variant)
            for b in branches:branch_counts[b]+=1
            changed_total+=changed;footprint_total+=footprints
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=predict(model,x)
            loss,geo,keep=objective(out,x,y,v,k,dc['keep_epsilon']);assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),C['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%50==0:
                row=dict(step=step,loss=float(loss.detach()),geometry=float(geo.detach()),keep=float(keep.detach()),gradient_norm=float(norm),elapsed_seconds=time.monotonic()-begin)
                log.write(json.dumps(row)+'\n');log.flush();curve.append(row)
                if step%250==0:print('STEP',variant,step,round(row['loss'],6),flush=True)
            if step in C['evaluation_steps']:
                ts=time.monotonic();splits=['pilot_train','pilot_dev'] if step==C['steps'] else ['pilot_dev']
                evaluation=evaluate(model,cache,scenes,splits,output=dest if step==C['steps'] else None)
                save_json(dest/f'eval-{step:05d}.json',evaluation);eval_seconds+=time.monotonic()-ts
                print('EVAL',variant,step,json.dumps(quick_summary(evaluation)),flush=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-begin
    final_hash=model_hash(model);assert final_hash!=init
    torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),config=C,
                    variant=variant,step=C['steps'],init_sha256=init,schedule_sha256=plan_hash),dest/'final.pt')
    save_json(dest/'complete.json',dict(variant=variant,steps=C['steps'],exposures=sum(branch_counts),branch_counts=branch_counts,
        actual_perturbation_changed_crop_pixels=changed_total,perturbation_footprint_crop_pixels=footprint_total,
        init_sha256=init,final_sha256=final_hash,schedule_sha256=plan_hash,parameters=sum(p.numel() for p in model.parameters()),
        training_and_eval_seconds=elapsed,evaluation_seconds=eval_seconds,training_loop_seconds_excluding_eval=elapsed-eval_seconds,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,final_dev_natural=quick_summary(evaluation),
        final_dev_perturbed=quick_summary(evaluation,branch='perturbed'),curve=curve))
    print('COMPLETE',variant,round(elapsed,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','train']);p.add_argument('--variant',choices=C['variants']);a=p.parse_args()
    smoke() if a.phase=='smoke' else train(a.variant)
