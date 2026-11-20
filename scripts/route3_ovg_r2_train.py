"""Fixed 30,000-step second-round training with a growing shard inventory."""
import argparse
from collections import Counter
import json
import os
import subprocess
import time

import numpy as np
import torch

from route3_ovg_r2_common import CONFIG, OUT, ROUND1, guard, reference
import route3_ovg_data as data
import route3_ovg_train as base
from route3_ovg_subpixel import save_json
from src.ovg_net import geometry_loss, OVGNet


def setup_data():
    reference(ROUND1/'B',OUT/'B')
    reference(ROUND1/'C/mesh-check.json',OUT/'C/mesh-check.json')
    for tier in ('G','C'):
        for source in (ROUND1/tier/'windows').glob('*/manifest.json'):
            reference(source.parent,OUT/tier/'windows'/source.parent.name)
    base.OUT=data.OUT=OUT


def model(device='cuda'):
    first=base.make('cpu')
    net=OVGNet(first.encoder,learned_prior_selection=True).to(device)
    return net


def step(net,opt,batch,rng):
    x,y=batch
    opt.zero_grad(set_to_none=True)
    out=net(x)
    loss,t,s=geometry_loss(out,x,y,base.CONFIG['weights'],
                        boundary_sampling=(CONFIG['per_stratum_pixels'],CONFIG['boundary_weight']),generator=rng)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite stage 1a objective')
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
    opt.step()
    return dict(total=float(loss.detach()),grad_norm=float(grad),**{k:float(v.detach()) for k,v in t.items()},
                **{k+'_support':int(v) for k,v in s.items()},local_pixels=int(out['mv_mask'].sum()))


def seeded():
    torch.manual_seed(CONFIG['seed'])
    return torch.Generator(device='cuda').manual_seed(CONFIG['seed']+17)


def check():
    path=OUT/'single-batch.json'
    if path.exists() and json.loads(path.read_text())['passed']:
        return
    rng=seeded()
    net=model().train()
    B=next(x for x in data.inventory('train')['B'] if data.TRAIN_FRAMES[x[2]]==30)
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=base.CONFIG['learning_rate'])
    batch=data.sample(B,'cuda')
    start=time.monotonic()
    curve=[dict(step=i+1,**step(net,opt,batch,rng)) for i in range(base.CONFIG['single_batch_steps'])]
    with torch.no_grad():
        out=net(batch[0])
    weights=out['prior_weights']
    present=batch[0]['prior_conf']>0
    assert (weights[~present]==0).all()
    assert torch.allclose(weights.sum(1),present.any(1).float(),atol=1e-6)
    assert curve[-1]['dual_support']>0 and all(p.grad is None for p in net.encoder.parameters())
    passed=curve[-1]['total']<curve[0]['total']
    save_json(path,dict(passed=passed,sample=list(B),frame=30,architecture=base.info(net),curve=curve,
                       masking='missing prior weight exactly zero; all-missing strategy has zero weights and 5 m base',
                       failure_action='inspect supervised-pixel sampling, masks and loss; reread only affected check',
                       seconds=time.monotonic()-start))
    print('ROUND2_B_CHECK',passed,curve[0]['total'],curve[-1]['total'],flush=True)
    if not passed:
        raise RuntimeError('Real B fixed-batch loss did not descend')


def corpus(sets):
    def env(x):
        return json.loads(open(x[1]).read())['environment'] if x[0]=='G' else None
    return dict(sizes={t:len(v) for t,v in sets.items()},
                G_environments=dict(sorted(Counter(env(x) for x in sets['G']).items())),
                C_sequences=len({x[1] for x in sets['C']}),
                B_scene_names=sorted({x[1] for x in sets['B']}))


def groups_G(sets):
    by={}
    cache={}
    for item in sets['G']:
        path=item[1]
        if path not in cache:
            with open(path) as f:
                cache[path]=json.load(f)['environment']
        by.setdefault(cache[path],[]).append(item)
    return [v for k,v in sorted(by.items())]


def train():
    check_state=OUT/'single-batch.json'
    if not check_state.exists() or not json.loads(check_state.read_text())['passed']:
        raise RuntimeError('B batch check must pass')
    steps=CONFIG['train_steps']
    done=OUT/'training.json'
    if done.exists():
        return
    generator=seeded()
    rng=np.random.default_rng(CONFIG['seed'])
    net=model()
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=base.CONFIG['learning_rate'])
    checkpoint_dir=OUT/'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    start=1
    counts=Counter()
    environments=Counter()
    curve,joins,evaluations=[],[],[]
    existing=sorted(checkpoint_dir.glob('step-*.pt'))
    if existing:
        ck=torch.load(existing[-1],weights_only=True,map_location='cpu')
        net.load_state_dict(ck['model'],strict=False)
        opt.load_state_dict(ck['optimizer'])
        generator.set_state(ck['pixel_rng'])
        rng.bit_generator.state=ck['sampling_rng']
        start=ck['step']+1
        history=json.loads((OUT/'checkpoint-history'/f'step-{ck["step"]:06d}.json').read_text())
        counts.update(history['counts'])
        environments.update(history['G_sampled_environments'])
        curve,joins,evaluations=history['curve'],history['joins'],history['evaluations']
    probs=CONFIG['batch_probability']
    pool=data.inventory('train')
    gids=groups_G(pool)
    for k in probs:
        if not pool[k]:
            raise FileNotFoundError(f'No valid training item for {k}')
    watch={5000,10000,15000,20000,25000,30000}
    assert watch==set(CONFIG['checkpoint_steps'])
    clock=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for i in range(start,steps+1):
        if i==start or (i-1)%100==0:
            guard()
            pool=data.inventory('train')
            gids=groups_G(pool)
            sizes={k:len(v) for k,v in pool.items()}
            if not joins or sizes!=joins[-1]['sizes']:
                joins.append(dict(step=i,**corpus(pool)))
        tier=str(rng.choice(list(probs),p=list(probs.values())))
        choices=gids[int(rng.integers(len(gids)))] if tier=='G' else pool[tier]
        item=choices[int(rng.integers(len(choices)))]
        with torch.enable_grad():
            net.train()
            try:
                values=step(net,opt,data.sample(item,'cuda'),generator)
            except (FloatingPointError,RuntimeError,ValueError) as error:
                save_json(OUT/'training-error.json',dict(step=i,tier=tier,item=list(item),error=str(error)))
                raise
        counts[tier]+=1
        if tier=='G':
            environments[item[1].split('/')[-2].rsplit('-',2)[0]]+=1
        curve.append(dict(step=i,tier=tier,**values))
        if i==1 or i%100==0:
            save_json(OUT/'progress.json',dict(step=i,counts=counts,latest=curve[-1],joins=joins))
            print('OVG_R2',i,tier,round(values['total'],5),dict(counts),joins[-1]['sizes'],flush=True)
        if i in watch:
            evaluation=base.evaluate(net,i,'cuda',i==steps)
            path=checkpoint_dir/f'step-{i:06d}.pt'
            state={k:v.detach().cpu() for k,v in net.state_dict().items() if not k.startswith('encoder.')}
            row=next(r for r in evaluation['paired'] if r['scene']=='pooled' and r['prior']=='DepthSplat')
            evaluations.append(dict(step=i,checkpoint=str(path.relative_to(data.ROOT)),frames=evaluation['frames'],
                                    boundary_delta_vs_DepthSplat_px=row['delta_boundary_front_CoC_px']))
            save_json(OUT/'checkpoint-history'/f'step-{i:06d}.json',dict(counts=counts,curve=curve,joins=joins,
                          evaluations=evaluations,G_sampled_environments=environments))
            tmp=path.with_suffix('.part')
            torch.save(dict(step=i,model=state,optimizer=opt.state_dict(),sampling_rng=rng.bit_generator.state,
                            pixel_rng=generator.get_state(),architecture='learned_prior_selection',
                            config=CONFIG,encoder=base.WEIGHT.name),tmp)
            tmp.replace(path)
    save_json(done,dict(steps=steps,counts=counts,proportions={k:counts[k]/steps for k in probs},
              curve=curve,joins=joins,evaluations=evaluations,architecture=base.info(net),
              seconds=time.monotonic()-clock,G_sampled_environments=environments,
              peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,torch=str(torch.__version__),
              starting_parameters='new geometry heads, same frozen DINOv2 weight as round 1; no round-1 network warm start',
              GT_warmup=False,selection='fixed last step; validation reads scores only',
              controls=dict(boundary_weight=CONFIG['boundary_weight'],samples_per_stratum=CONFIG['per_stratum_pixels'],
                            optimizer='round1 AdamW and learning rate, no change from dev scores')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['check','train'])
    a=p.parse_args()
    guard()
    setup_data()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    subprocess.run(['nvidia-smi'],check=True)
    if torch.cuda.mem_get_info()[0]<24*2**30:
        raise RuntimeError('Concurrent GPU budget unavailable')
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('high')
    {'check':check,'train':train}[a.stage]()
