"""Verified single-batch check and mixed-tier OVG-Net stage 1a."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from route3_ovg_data import CONFIG, OUT, ROOT, TRAIN_FRAMES, PRIORS, b_edge_reference, inventory, sample
from route3_ovg_subpixel import save_json, save_npz
from src.ovg_net import OVGNet, geometry_loss
from src.ovg_edge_statistics import EdgeStatistics

DINO=ROOT.parent/'dinov2-depthsplat-r13'
WEIGHT=ROOT/'models/OVG-Net/dinov2_vits14_pretrain.pth'


def guard():
    if shutil.disk_usage(OUT).free/2**30<60:
        raise RuntimeError('Data disk below 60 GiB')


def make(device):
    sys.path.insert(0,str(DINO))
    from dinov2.hub.backbones import dinov2_vits14
    encoder=dinov2_vits14(pretrained=False)
    encoder.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
    return OVGNet(encoder).to(device)


def info(net):
    return dict(total=sum(x.numel() for x in net.parameters()),
                trainable=sum(x.numel() for x in net.parameters() if x.requires_grad),
                frozen_encoder=sum(x.numel() for x in net.encoder.parameters()),
                ViT='DINOv2 ViT-S/14, 384 channels, output blocks 2/5/8/11, 252x252 token input',
                CNN='1/2:32, 1/4:48, separate prior-motion CNN, four-scale DPT and 64-channel per-pixel decoder',
                multiview='two neighboring input frames; sparse quarter-grid eight inverse-depth hypotheses, input depth confidence and edge mask minus motion',
                outputs=['mu1','mu2','alpha','normal','confidence','sigma_px two axes','opacity'],
                footprint='head exposed, no 1a raster/coverage loss so not calibrated',
                appearance_head_enabled=False,
                encoder_revision=subprocess.check_output(['git','-C',str(DINO),'rev-parse','HEAD'],text=True).strip(),
                weight=str(WEIGHT.relative_to(ROOT)),license='Apache-2.0',
                source='https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md')


def optimization(net,optimizer,inputs,labels):
    optimizer.zero_grad(set_to_none=True)
    out=net(inputs)
    loss,terms,masks=geometry_loss(out,inputs,labels,CONFIG['weights'])
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite OVG-Net training loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    return dict(total=float(loss.detach()),grad_norm=float(norm.detach()),
                **{k:float(x.detach()) for k,x in terms.items()},
                support={k:int(x.item()) for k,x in masks.items()},
                local_pixels=int(out['mv_mask'].sum()))


def batch_check(device):
    dest=OUT/'W1-single-batch.json'
    if dest.exists():
        return
    examples=inventory('train')['B']
    b=next(item for item in examples if TRAIN_FRAMES[item[2]]==30)
    net=make(device).train()
    P=info(net)
    inputs,labels=sample(b,device)
    assert 'depth1' not in inputs and 'alpha' not in inputs and labels['valid_dual'].any()
    opt=torch.optim.AdamW([x for x in net.parameters() if x.requires_grad],lr=CONFIG['learning_rate'])
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    path=[]
    for step in range(CONFIG['single_batch_steps']):
        path.append(dict(step=step+1,**optimization(net,opt,inputs,labels)))
    assert all(p.grad is None for p in net.encoder.parameters())
    decrease=path[-1]['total']<path[0]['total']
    save_json(dest,dict(passed=decrease,sample=list(b),actual_frame=TRAIN_FRAMES[b[2]],
                    curve=path,criterion='finite forward/backward and last direct loss < first on fixed real B batch',
                    decision='pass: launch once G/C/B ready; otherwise inspect loader and masks',
                    architecture=P,seconds=time.monotonic()-start,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
    print('B_FIXED_BATCH',path[0]['total'],path[-1]['total'],decrease,flush=True)
    if not decrease:
        raise RuntimeError('B fixed batch failed decrease; inspect masks and loader')


def measure(prediction, alpha, rear, gt, valid, factor, boundary, interior):
    vv=valid & gt['valid']
    front_error=(prediction.clamp_min(.05).reciprocal()-gt['depth1'].clamp_min(.05).reciprocal()).abs()
    rear_mask=gt['valid_dual']&vv if rear is not None else vv&False
    a_mask=gt['valid_alpha']&vv if alpha is not None else vv&False
    alpha_edge=a_mask & gt['depth_boundary']
    int_mask=a_mask&interior
    return dict(front_n=int(vv.sum()),front_sum=float(front_error[vv].sum()),
                front_coc_sum=float((front_error*factor)[vv].sum()),
                edge_front_n=int((vv&boundary).sum()),edge_front_sum=float(front_error[vv&boundary].sum()),
                edge_front_coc_sum=float((front_error*factor)[vv&boundary].sum()),
                edge_back_n=int(rear_mask.sum()),edge_back_sum=float(
                    (rear.clamp_min(.05).reciprocal()-gt['depth2'].clamp_min(.05).reciprocal()).abs()[rear_mask].sum()
                ) if rear is not None else 0.,
                edge_back_coc_sum=float(((rear.clamp_min(.05).reciprocal()-gt['depth2'].clamp_min(.05).reciprocal()).abs()*factor)[rear_mask].sum()
                ) if rear is not None else 0.,
                alpha_n=int(a_mask.sum()),alpha_sum=float((alpha-gt['alpha']).abs()[a_mask].sum()) if alpha is not None else 0.,
                alpha_edge_n=int(alpha_edge.sum()),
                alpha_edge_sum=float((alpha-gt['alpha']).abs()[alpha_edge].sum()) if alpha is not None else 0.,
                interior_n=int(int_mask.sum()),false_transparent_n=int(((alpha<.95)&int_mask).sum()) if alpha is not None else 0.)


def reduce_metrics(v):
    div=lambda s,n:v[s]/v[n] if v[n] else None
    return dict(**v,front_inverse_mae=div('front_sum','front_n'),front_CoC_mae_px=div('front_coc_sum','front_n'),
                boundary_front_inverse_mae=div('edge_front_sum','edge_front_n'),
                boundary_front_CoC_mae_px=div('edge_front_coc_sum','edge_front_n'),
                boundary_back_inverse_mae=div('edge_back_sum','edge_back_n'),
                boundary_back_CoC_mae_px=div('edge_back_coc_sum','edge_back_n'),
                coverage_MAE=div('alpha_sum','alpha_n'),boundary_coverage_MAE=div('alpha_edge_sum','alpha_edge_n'),
                false_semitransparent=div('false_transparent_n','interior_n'))


@torch.no_grad()
def evaluate(net,step,device,save_images,*,dev_items=None,sample_item=sample):
    net.eval()
    entries=inventory('dev')['B'] if dev_items is None else dev_items
    edge_stats=EdgeStatistics()
    results=[]
    selection=defaultdict(Counter)
    for item in entries:
        x,y=sample_item(item,device)
        out=net(x)
        if not all(torch.isfinite(v).all().item() for v in [out['depth1'],out['depth2'],out['alpha'],out['normal']]):
            raise FloatingPointError(f'Nonfinite validation geometry {item}')
        frame=TRAIN_FRAMES[item[2]]
        factor=x['K'][0,0,0]*.05/(2*1.4)
        # Segregate the strict two-mode boundary from opaque interior.
        edge=y['depth_boundary'] & y['valid']
        interior=y['valid_alpha']&(y['alpha']>=.9999)&~edge
        v=y['valid']
        base_name='selected-prior-base' if 'prior_weights' in out else 'confidence-fusion'
        methods={'OVG-Net':(out['depth1'],out['alpha'],out['depth2'],v,'all'),
                 base_name:(out['fused_depth'],None,None,x['prior_conf'].sum(1)>0,'supported')}
        if 'prior_weights' in out:
            weights=out['prior_weights']
            supported=weights.sum(1)>0
            selected=weights.argmax(1)
            if (weights[x['prior_conf']<=0]!=0).any() or not torch.isfinite(weights).all():
                raise ValueError('Invalid masked prior weights')
            for tag,part in [('interior',interior),('boundary',edge),('reliable_dual',y['valid_dual'])]:
                use=part&supported
                counts=torch.bincount(selected[use],minlength=3).cpu().tolist()
                mass=(weights*use[:,None]).sum((0,2,3)).cpu().tolist()
                for group in (item[1],'pooled'):
                    acc=selection[group,tag]
                    acc.update(total=int(part.sum()),supported=int(use.sum()))
                    acc.update({p+'_wins':counts[j] for j,p in enumerate(PRIORS)})
                    acc.update({p+'_mass':mass[j] for j,p in enumerate(PRIORS)})
        for k,prior in enumerate(PRIORS):
            supported=x['prior_conf'][:,k]>0
            methods[prior]=(x['prior_depth'][:,k],None,None,supported,'supported')
            methods['OVG on '+prior]=(out['depth1'],out['alpha'],out['depth2'],supported,'paired')
        for name,(z,a,back,m,domain) in methods.items():
            r=measure(z,a,back,y,m,factor,edge,interior)
            results.append(dict(scene=item[1],frame=frame,method=name,domain=domain,**r))
        reference=b_edge_reference(item,y)
        for name,(z,_,_,m,_) in methods.items():
            edge_stats.add(item[1],name,z[0].cpu().numpy(),reference,m[0].cpu().numpy())
        if save_images:
            outpath=OUT/'dev-predictions'/item[1]/f'f{frame:03d}.npz'
            save_npz(outpath,**{k:val[0].cpu().numpy().astype(np.float16)
                                for k,val in out.items() if k!='mv_mask'},mv_mask=out['mv_mask'][0].cpu().numpy())
    agg=defaultdict(Counter)
    datafields=set(results[0])-{'scene','frame','method','domain'}
    for row in results:
        for group in (row['scene'],'pooled'):
            agg[group,row['method'],row['domain']].update({k:row[k] for k in datafields})
    pooled=[dict(scene=s,method=m,domain=domain,**reduce_metrics(stats))
            for (s,m,domain),stats in sorted(agg.items())]
    by={(r['scene'],r['method']):r for r in pooled}
    paired=[]
    for s in sorted({x['scene'] for x in pooled}):
        for p in PRIORS:
            a,b=by[s,p],by[s,'OVG on '+p]
            assert a['front_n']==b['front_n']
            paired.append(dict(scene=s,prior=p,pixels=a['front_n'],
                delta_inverse_mae=(b['front_inverse_mae']-a['front_inverse_mae']) if a['front_n'] else None,
                delta_CoC_px=(b['front_CoC_mae_px']-a['front_CoC_mae_px']) if a['front_n'] else None,
                delta_boundary_front_CoC_px=(b['boundary_front_CoC_mae_px']-a['boundary_front_CoC_mae_px'])
                if a['edge_front_n'] else None))
    record=dict(step=step,frames=len(entries),pooled=pooled,paired=paired,per_frame=results,
        edge_inverse_depth=list(edge_stats.rows()),
        coverage_definition='geometry coverage, only valid labels; false semi-transparent: predicted alpha < .95 on solid GT interior',
        prior_coverage_and_second_layer='single-depth prior has neither explicit quantity; blank',
        paired_definition='each raw prior and model use exactly that prior confidence>0 and GT-valid pixels; negative delta improves',
        CoC='fx*.05/(2*1.4) times inverse axial depth error; native B 512 pixel units',
        resolution='B 512; exact native subpixel labels',
        data_split='three development scenes, never sampled by optimizer')
    if selection:
        record['prior_selection']=[
            dict(scene=scene,region=tag,**acc,
                 **{p+'_vote_fraction':acc[p+'_wins']/acc['supported'] if acc['supported'] else None for p in PRIORS},
                 **{p+'_mean_weight':acc[p+'_mass']/acc['supported'] if acc['supported'] else None for p in PRIORS})
            for (scene,tag),acc in sorted(selection.items())]
        record['prior_selection_definition']='argmax and mean softmax weight among pixels with any prior; missing prior has exact zero, unsupported counts separate'
    save_json(OUT/'validation'/f'step-{step:06d}.json',record)
    return record


def checkpoint(net,opt,step,rng):
    root=OUT/'checkpoints';root.mkdir(exist_ok=True)
    p=root/f'step-{step:06d}.pt'
    temp=p.with_suffix('.part')
    weights={k:v.detach().cpu() for k,v in net.state_dict().items() if not k.startswith('encoder.')}
    torch.save(dict(step=step,trainable_model=weights,optimizer=opt.state_dict(),
                    rng=rng.bit_generator.state,backbone_reference=str(WEIGHT.relative_to(ROOT)),
                    seed=CONFIG['seed'],config=CONFIG),temp)
    temp.replace(p)
    return p


def train(device):
    checks=OUT/'W1-single-batch.json'
    if not checks.exists() or not json.loads(checks.read_text())['passed']:
        raise RuntimeError('The required real B batch check has not passed')
    data=inventory('train')
    minimum=CONFIG['minimum_train_samples']
    for tier,n in minimum.items():
        if len(data[tier])<n:
            raise FileNotFoundError(f'{tier} valid shards have not reached configured minimum')
    net=make(device)
    opt=torch.optim.AdamW([x for x in net.parameters() if x.requires_grad],lr=CONFIG['learning_rate'])
    rng=np.random.default_rng(CONFIG['seed'])
    probs=CONFIG['batch_probability']
    kinds=list(probs)
    counts=Counter()
    curve,joins,checks=[],[],[]
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    for step in range(1,CONFIG['train_steps']+1):
        if step==1 or (step-1)%100==0:
            guard()
            data=inventory('train')
            current={k:len(data[k]) for k in kinds}
            if not joins or joins[-1]['sizes']!=current:
                joins.append(dict(step=step,sizes=current))
        tier=str(rng.choice(kinds,p=list(probs.values())))
        example=data[tier][int(rng.integers(len(data[tier])))]
        x,y=sample(example,device)
        net.train()
        try:
            row=optimization(net,opt,x,y)
        except (FloatingPointError,RuntimeError) as error:
            save_json(OUT/'1a-training-error.json',dict(step=step,sample=list(example),error=str(error)))
            raise
        counts[tier]+=1
        curve.append(dict(step=step,tier=tier,scene=Path(example[1]).parent.name if tier!='B' else example[1],**row))
        if step%50==0:
            print('OVG_TRAIN',step,tier,row['total'],counts,flush=True)
            save_json(OUT/'1a-progress.json',dict(step=step,counts=counts,recent=curve[-50:]))
        if step in CONFIG['checkpoint_steps']:
            path=checkpoint(net,opt,step,rng)
            record=evaluate(net,step,device,step==CONFIG['train_steps'])
            checks.append(dict(step=step,checkpoint=str(path.relative_to(ROOT)),frames=record['frames']))
    save_json(OUT/'1a-training.json',dict(steps=step,counts=counts,
             proportions={k:counts[k]/step for k in kinds},curve=curve,validations=checks,
             joins=joins,seconds=time.monotonic()-start,
             peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,seed=CONFIG['seed'],
             architecture=info(net),torch=str(torch.__version__),GPU=torch.cuda.get_device_name(0),
             no_GT_geometry_warmup=True,two_stage_training='only 1a geometry evaluation',
             per_tier_losses='L_disp/L_normal/L_mv on G and C/B; L_bimodal/L_alpha only on labelled C/B positions'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['check','train'])
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    guard()
    subprocess.run(['nvidia-smi'],check=True)
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(CONFIG['seed'])
    {'check':batch_check,'train':train}[args.stage]('cuda')
