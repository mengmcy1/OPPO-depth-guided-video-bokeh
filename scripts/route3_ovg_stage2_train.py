"""Fixed-length appearance curriculum; frozen 1c geometry and Hidden Query."""
import argparse
from collections import Counter
from functools import lru_cache
import os
import shutil
import subprocess
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import route3_ovg_data as data
from route3_ovg_1b_common import CONFIG, OUT as DATA_RUN, ROOT, read
from route3_ovg_1b_optical_data import background_samples, occupancy_labels, read_optical, synthetic_optical
from route3_ovg_1b_train import load_model
from route3_ovg_1c import OUT as RUN1C, aligned_sample
from route3_ovg_phase2_labels import image as rgba
from route3_ovg_stage2_data import OUT, SIZE, SEED, compose
from route3_ovg_stage2_query import FrozenQuery, Q_MIN, present_rows
from route3_ovg_subpixel import save_json, save_npz
from route3_upper_bound_evaluate import display
from src.layered_aperture import merge_layers
from src.ovg_appearance import AppearanceHeads, boundary_candidate, boundary_layer, cutoff, visible_layer
from src.ovg_net import geometry_loss, masked_mean
from src.ovg_optics import fixed_visible_surfels, occlusion_loss, raster, stratified_pupil, subcamera
from src.prefix_transmittance import differentiable_prefix_transmittance

INITIAL=RUN1C/'train/aligned/checkpoints/step-005000.pt'
B_LABELS=RUN1C/'phase2-B'
STAGES={'2a':(10000,dict(A=.70,C=.30)), '2b':(30000,dict(C=.40,B=.40,A=.20))}
WEIGHTS=dict(CONFIG['loss_weights'],foreground=1.,comp=1.,coverage=.5)


def guard(gpu=False):
    OUT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(OUT).free<60*2**30:
        raise RuntimeError('Data disk below 60 GiB')
    if gpu:
        subprocess.run(['nvidia-smi'],check=True)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
            raise RuntimeError('Expected assigned GPU 0')
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        data.OUT=DATA_RUN


def tensor(v):
    return torch.as_tensor(v,device='cuda')


def original_B(name,frame):
    return ('B',name,data.TRAIN_FRAMES.index(frame))


def sampled(item):
    return aligned_sample(item,'cuda') if item[0]=='B' else data.sample(item,'cuda')


def output_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def load_checkpoint(net, path, ema=True):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    net.load_state_dict(ck['ema' if ema else 'model'])
    return ck


def small_fields(x,geom):
    size=geom['depth1'].shape[-1]
    if size==SIZE:
        return x,geom
    K=x['K'].clone()
    K[:,:2]*=SIZE/size
    x_small=dict(x,rgb=F.interpolate(x['rgb'],(SIZE,SIZE),mode='area'),K=K)
    indices={name: F.interpolate(geom[name][:,None],(SIZE,SIZE),mode='nearest-exact')[:,0]
             for name in ('depth1','depth2')}
    return x_small,dict(indices,features=F.interpolate(geom['features'],(SIZE,SIZE),
                            mode='bilinear',align_corners=False))


def alpha_plane(rgb,alpha):
    return torch.where(alpha[...,None]>0,rgb/alpha[...,None].clamp_min(1e-8),0.)


@lru_cache(maxsize=24)
def label_appearance(tier,key,index):
    if tier=='C':
        scene=read(key)['scene']
        d=data.load(OUT/'C'/scene/f'f{index:03d}-appearance.npz')
        return dict(alpha=d['alpha'],foreground=d['foreground'],background=d['background'],
                    color_valid=(d['alpha']>0)&(d['alpha']<1),comp_valid=(d['alpha']>0)&(d['alpha']<1),
                    full=d['full'])
    frame=data.TRAIN_FRAMES[index]
    folder=B_LABELS/key
    cov=data.load(folder/f'f{frame:03d}-coverage.npz')
    f=data.load(folder/f'f{frame:03d}-foreground-label.npz')
    path=folder/f'f{frame:03d}-visible-background.exr'
    if path.is_file():
        b=rgba(path)
        bg=np.divide(b[...,:3],b[...,3,None],out=np.zeros_like(b[...,:3]),where=b[...,3,None]>0)
        valid=(b[...,3]>0)&f['valid_boundary_color']
    else:
        bg=np.zeros_like(f['linear_rgb'])
        valid=np.zeros(cov['coverage'].shape,bool)
    return dict(alpha=cov['coverage'],foreground=f['linear_rgb'],background=bg,
                color_valid=f['valid_boundary_color'],comp_valid=valid)


class Model:
    def __init__(self):
        self.backbone=load_model(INITIAL,'ema').eval().requires_grad_(False)
        self.heads=AppearanceHeads(self.backbone.coverage).cuda()
        self.query=FrozenQuery()

    @torch.no_grad()
    def geometry(self,x):
        return self.backbone(x,protect_residual=True,render_heads=False,return_features=True)

    @lru_cache(maxsize=12)
    def first_record(self,item,size=SIZE):
        tier,key,index=item
        name=key if tier=='B' else read(key)['scene']
        path=OUT/'observed-cache'/f'{size}'/tier/name/f'{index:03d}.npz'
        if not path.exists():
            x,_=sampled(item)
            g=self.geometry(x)
            if size==SIZE:
                _,g=small_fields(x,g)
            layers=self.query.first(tier,key,index,g['depth1'][0].cpu().numpy(),g['depth2'][0].cpu().numpy())
            save_npz(path,**layers)
        return data.load(path)

    @torch.no_grad()
    def observations(self,item,native=False):
        x,y=sampled(item)
        if native:
            g=self.geometry(x)
            return x,y,g,self.first_record(item,512)
        tier,key,index=item
        scene=key if tier=='B' else read(key)['scene']
        path=OUT/'frozen-geometry'/tier/scene/f'{index:03d}.npz'
        if not path.exists():
            x,g=small_fields(x,self.geometry(x))
            save_npz(path,depth1=g['depth1'].cpu().numpy(),depth2=g['depth2'].cpu().numpy(),
                     features=g['features'].cpu().numpy().astype(np.float16))
        else:
            if x['rgb'].shape[-1]!=SIZE:
                K=x['K'].clone()
                K[:,:2]*=SIZE/x['rgb'].shape[-1]
                x=dict(x,K=K,rgb=F.interpolate(x['rgb'],(SIZE,SIZE),mode='area'))
            g={k:tensor(v).float() for k,v in data.load(path).items()}
        g['inverse1']=g['depth1'].reciprocal()
        g['inverse2']=g['depth2'].reciprocal()
        g['fused_depth']=g['depth1']
        return x,y,g,self.first_record(item,SIZE)

    def appearance(self,g,x,first):
        color=tensor(first['memory']).permute(2,0,1)[None]
        confidence=tensor(first['memory_q'])[None]
        return self.heads(g,x['rgb'],color,confidence)

    @lru_cache(maxsize=8)
    def f2_entry(self,name,frame):
        root=data.PREVIOUS/'geometry/original'/name
        return next((r for r in read(root/'prepared.json')['rows'] if r['frame']==frame),None)

    @lru_cache(maxsize=12)
    def second(self,item,spec_focus,aperture):
        x,y,g,first=self.observations(item)
        tier,key,index=item
        return self.query.second(tier,key,index,g['depth1'][0].cpu().numpy(),spec_focus,aperture)


def a_item(model,assets,rng,simulate):
    choices=[assets[k][int(rng.integers(len(assets[k])))] for k in ('foreground','background')]
    rgb,fg,alpha,bg,mem,valid=compose(*(np.load(ROOT/v) for v in choices),rng,simulate)
    xrgb=tensor(rgb.transpose(2,0,1)[None])
    h,w=alpha.shape
    z=xrgb.new_zeros((1,3,h,w))
    x=dict(rgb=xrgb,prior_depth=z,prior_conf=z.clone(),motion=tensor(np.zeros((1,h,w),bool)),
           neighbors=xrgb.new_zeros((1,0,3,h,w)))
    g=model.geometry(x)
    o=model.appearance(g,x,dict(memory=mem,memory_q=valid.astype(np.float32)))
    a=tensor(alpha)
    edge=(a>0)&(a<1)
    f=o['foreground'][0].permute(1,2,0)
    rgb_hat=a[...,None]*f+(1-a[...,None])*tensor(bg)
    terms=dict(foreground=masked_mean((f-tensor(fg)).abs().mean(-1),edge),
               comp=masked_mean((rgb_hat-tensor(rgb)).abs().mean(-1),edge))
    return sum(WEIGHTS[k]*v for k,v in terms.items()),terms,dict(b=None,valid_pixels=int(edge.sum()))


def label_small(label):
    res={}
    for k,v in label.items():
        if v.ndim==3:
            res[k]=F.interpolate(v[:,None].float(),(SIZE,SIZE),mode='nearest-exact')[:,0].to(v.dtype)
        else:
            res[k]=F.interpolate(v,(SIZE,SIZE),mode='nearest-exact')
    return res


def b_or_c(model,item,rng,generator):
    tier,key,index=item
    original_x,_=sampled(item)
    x,y,g,first=model.observations(item)
    label=label_small(y) if tier=='B' else y
    ap=model.appearance(g,x,first)
    split=boundary_candidate(ap)&(tensor(first['q'])>=Q_MIN)
    rendered=visible_layer(ap,x,split)
    first_bg=boundary_layer(ap,x,split,tensor(first['color']),tensor(first['depth']))
    K=x['K'][0]
    center=K.new_zeros((1,2))
    intr,view=subcamera(K,1.4,center[0])
    T=differentiable_prefix_transmittance(rendered,intr,view,cutoff(ap['depth1'][0],ap['depth2'][0]))
    cut=1-T
    terms={'coverage':masked_mean((cut-ap['alpha'][0]).abs(),label['valid_alpha'][0])}

    # Independent color targets remain confined to supervision.
    appearance_target=label_appearance(tier,key,index)
    def local(field):
        a=appearance_target[field]
        if a.shape[0]!=SIZE:
            a=cv2.resize(a.astype(np.uint8) if a.dtype==bool else a,(SIZE,SIZE),
                         interpolation=cv2.INTER_NEAREST_EXACT)
            if field.endswith('_valid'):
                a=a.astype(bool)
        return tensor(a)
    f=ap['foreground'][0].permute(1,2,0)
    a=ap['alpha'][0][...,None]
    rgb=x['rgb'][0].permute(1,2,0)
    comp_reference=local('full') if tier=='C' else local('foreground')*local('alpha')[...,None]+(
        1-local('alpha')[...,None])*local('background')
    if tier=='B':
        # Use the captured pixel I on the same sample footprint as the matte.
        comp_reference=F.interpolate(original_x['rgb'],(SIZE,SIZE),mode='nearest-exact')[0].permute(1,2,0)
    terms['foreground']=masked_mean((f-local('foreground')).abs().mean(-1),local('color_valid'))
    terms['comp']=masked_mean((a*f+(1-a)*local('background')-comp_reference).abs().mean(-1),local('comp_valid'))
    # Retain the phase-1 losses on their original native label/camera lattice.
    native_geo=model.geometry(original_x) if tier=='B' else g
    native_ap=dict(native_geo)
    for k in ('alpha','normal'):
        result=ap[k]
        if tier=='B':
            result=F.interpolate(result[:,None] if result.ndim==3 else result,
                                 original_x['rgb'].shape[-2:],mode='nearest')
            if k=='alpha':
                result=result[:,0]
        native_ap[k]=result
    _,geometry_terms,_=geometry_loss(native_ap,original_x,y,WEIGHTS,
          boundary_sampling=CONFIG['boundary_sampling'],generator=generator,
          alpha_split_weights=CONFIG['alpha_interior_boundary_weights'])
    terms.update(geometry_terms)

    specs=synthetic_optical(item)
    spec=specs[int(rng.integers(len(specs)))] if specs else None
    focus=spec['focus'] if spec else 1.4
    frame=data.TRAIN_FRAMES[index] if tier=='B' else index
    entry=model.f2_entry(key,frame) if tier=='B' else None
    filler=dict(tiers='B',entry=entry) if entry is not None else None
    second=model.second(item,focus,spec['fstop']) if spec else None
    combined=merge_layers(merge_layers(rendered,first_bg),second)
    zrgb=raster(combined,K,focus,center,SIZE,SIZE,background_samples(filler,K,focus,center,(SIZE,SIZE)))[0]
    terms['zero']=(zrgb-rgb).abs().mean()
    if spec:
        offsets,cuts,reveal,block=occupancy_labels(spec,SIZE,SIZE,y,generator)
        terms['occ']=occlusion_loss(rendered,K,focus,offsets,cuts,reveal,block)
        ref=read_optical(spec)
        if ref is not None:
            radius=.05/(2*spec['fstop'])
            n=CONFIG['optical_train']['rays_per_subset']
            offsets=torch.cat([stratified_pupil(n,radius,generator,K) for _ in range(2)])
            colors=raster(combined,K,focus,offsets,SIZE,SIZE,background_samples(filler,K,focus,offsets,(SIZE,SIZE)))
            a,b=colors.chunk(2,dim=0)
            target=tensor(cv2.resize(ref,(SIZE,SIZE),interpolation=cv2.INTER_AREA))
            terms['bokeh']=((a.mean(0)-target)*(b.mean(0)-target)).mean()
    return sum(WEIGHTS[k]*v for k,v in terms.items()),terms,dict(b=float(split.float().mean()),
         valid_pixels=int(local('color_valid').sum()))


def gradient_norm(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


@torch.no_grad()
def monitor(model):
    rows=[]
    for name in read(data.PREVIOUS/'split.json')['heldout']:
        item=original_B(name,30)
        x,y,g,first=model.observations(item,native=True)
        p=model.appearance(g,x,first)
        select=boundary_candidate(p)&(tensor(first['q'])>=Q_MIN)
        visible=visible_layer(p,x,select)
        fg=boundary_layer(p,x,select,tensor(first['color']),tensor(first['depth']))
        K=x['K'][0]
        center=K.new_zeros((1,2))
        entry=model.f2_entry(name,30)
        spec=dict(tiers='B',entry=entry)
        second=model.second(item,1.4,1.4) if SIZE==g['depth1'].shape[-1] else model.query.second(
            'B',name,item[2],g['depth1'][0].cpu().numpy(),1.4,1.4)
        regions=dict(full=np.ones((512,512),bool), interior=(y['valid_alpha'][0]&~y['depth_boundary'][0]).cpu().numpy(),
                     boundary=y['depth_boundary'][0].cpu().numpy())
        rgb=display(x['rgb'][0].permute(1,2,0).cpu().numpy())
        row=dict(scene=name,b=float(select.float().mean()),
                 false_transparent=float((p['alpha'][0]<.95).cpu().numpy()[regions['interior']].mean()))
        full=merge_layers(merge_layers(visible,fg),second)
        for tag,sc,fill in (('visible',visible,None),
            ('combined',full,background_samples(spec,K,1.4,center,(512,512)))):
            sample=raster(sc,K,1.4,center,512,512,fill)
            errors=(display(sample[0].cpu().numpy())-rgb).astype(np.float64)**2
            row[tag]={k:float(-10*np.log10(max(errors[mask].mean(),1e-12))) for k,mask in regions.items()}
        rows.append(row)
    return rows


def smoke(model,assets):
    rng=np.random.default_rng(SEED)
    g=torch.Generator(device='cuda').manual_seed(SEED)
    names=data.inventory('dev')['B']
    item=next(v for v in names if '122105' in v[1] and data.TRAIN_FRAMES[v[2]]==30)
    x,_,geo,first=model.observations(item,native=True)
    with torch.no_grad():
        ap=model.appearance(geo,x,first)
        mask=boundary_candidate(ap)&(tensor(first['q'])>=Q_MIN)
        sc=visible_layer(ap,x,mask)
        original=fixed_visible_surfels(geo,x)
        for k in sc.__dataclass_fields__:
            torch.testing.assert_close(getattr(sc,k),getattr(original,k),rtol=0,atol=0)
        raw=raster(original,x['K'][0],1.4,x['K'].new_zeros((1,2)),512,512)[0]
        revised=raster(sc,x['K'][0],1.4,x['K'].new_zeros((1,2)),512,512)[0]
        diff=float((raw-revised).abs().max())
        if diff>1e-6:
            raise ValueError('Stage 2 fixed primitive identity failed')
        K=x['K'][0]
        center=K.new_zeros((1,2))
        spec=dict(tiers='B',entry=model.f2_entry(item[1],30))
        fallback=background_samples(spec,K,1.4,center,(512,512))
        second=model.query.second('B',item[1],item[2],geo['depth1'][0].cpu().numpy(),1.4,1.4)
        background=boundary_layer(ap,x,mask,tensor(first['color']),tensor(first['depth']))
        rgb=display(x['rgb'][0].permute(1,2,0).cpu().numpy())
        canvases=[merge_layers(original,second),merge_layers(merge_layers(sc,background),second)]
        scores=[]
        for canvas in canvases:
            sample=raster(canvas,K,1.4,center,512,512,fallback)[0]
            mse=np.square(display(sample.cpu().numpy()).astype(np.float64)-rgb).mean()
            scores.append(float(-10*np.log10(max(mse,1e-12))))
        if abs(scores[1]-scores[0])>.1:
            raise ValueError(f'Stage 2 full zero aperture initialization differs: {scores}')
        before={k:geo[k].clone() for k in ('depth1','depth2')}
    results=[]
    b=next(v for v in data.inventory('train')['B'] if synthetic_optical(v))
    for tier,task in (('A',lambda:a_item(model,assets,rng,True)),('B',lambda:b_or_c(model,b,rng,g))):
        model.heads.zero_grad(set_to_none=True)
        loss,terms,stats=task()
        loss.backward()
        amount=gradient_norm(model.backbone)
        alpha=gradient_norm(model.heads.coverage_delta)
        if amount or (tier=='A' and alpha) or (tier=='B' and not alpha):
            raise ValueError(f'Frozen geometry or tier gradient audit failed for {tier}')
        grad=float(torch.nn.utils.clip_grad_norm_(model.heads.parameters(),5,error_if_nonfinite=True))
        results.append(dict(tier=tier,loss=float(loss),terms={k:float(v) for k,v in terms.items()},
                            frozen_grad=amount,alpha_grad=alpha,trainable_grad_norm=grad,**stats))
    model.heads.zero_grad(set_to_none=True)
    for k,z in before.items():
        torch.testing.assert_close(model.geometry(x)[k],z,rtol=0,atol=0)
    save_npz(OUT/'S1-frozen-reference.npz',**{k:v.cpu().numpy() for k,v in before.items()})
    save_json(OUT/'S1.json',dict(passed=True,init_visible_max_linear_difference=diff,
             init_zero_PSNR_delta_db=scores[1]-scores[0],init_full_zero_PSNR_db=scores,
             init_boundary_b=int(mask.sum()),gradient_checks=results,identity_frame=list(item),
             trainable_parameters=sum(p.numel() for p in model.heads.parameters() if p.requires_grad),
             q_min=Q_MIN,confidence='selected head mean of learned sigmoid(q_logits) weighted by valid candidate selection',
             resolution_train=SIZE,scale='exp(tanh(delta)), same 5x5 centers, opacity 0.999+bounded correction on b only',
             geometry_and_shared_features='frozen 1c EMA and identical aligned input route',
             boundary_background_opacity='0.999 minus the learned front opacity; zero at initialization',weights=WEIGHTS))
    print('STAGE2_CHECK',results,flush=True)


def train():
    guard(gpu=True)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    assets=read(OUT/'A/manifest.json')
    model=Model()
    if not (OUT/'S1.json').exists():
        smoke(model,assets)
    if not read(OUT/'S0.json')['C_frames']:
        raise ValueError('Missing checked C labels')
    pool=data.inventory('train')
    pool['C']=[x for x in pool['C'] if (OUT/'C'/read(x[1])['scene']/f'f{x[2]:03d}-appearance.npz').exists()]
    if len(pool['B'])!=91 or not pool['C']:
        raise ValueError('Stage 2 train inventory invalid')
    opt=torch.optim.AdamW((p for p in model.heads.parameters() if p.requires_grad), lr=CONFIG['learning_rate'])
    rng=np.random.default_rng(SEED)
    generator=torch.Generator(device='cuda').manual_seed(SEED)
    ema={k:v.detach().clone() for k,v in model.heads.state_dict().items()}
    from route3_v2data_B_training_share import register_training
    for stage,(steps,prob) in STAGES.items():
        dest=OUT/stage
        dest.mkdir(exist_ok=True)
        register_training(stage,{stage:os.getpid()},{stage:str(dest/"progress.json")})
        curves,counts,uses=[],Counter(),Counter()
        elapsed=0.
        existing=sorted(dest.glob('step-*.pt'))
        start_step=1
        if existing:
            ck=load_checkpoint(model.heads,existing[-1],False)
            opt.load_state_dict(ck['optimizer'])
            for k in ema:
                ema[k].copy_(ck['ema'][k])
            rng.bit_generator.state=ck['rng']
            generator.set_state(ck['pixel_rng'])
            start_step=ck['step']+1
            curves=read(dest/f'step-{ck["step"]:06d}-curve.json')
            counts=Counter(r['tier'] for r in curves)
            uses=Counter(k for r in curves for k in r['terms'])
            elapsed=ck['seconds']
        start=time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step in range(start_step,steps+1):
            if step%100==1:
                guard()
            tier=str(rng.choice(list(prob),p=list(prob.values())))
            opt.zero_grad(set_to_none=True)
            if tier=='A':
                loss,terms,stats=a_item(model,assets,rng,stage=='2a')
            else:
                item=pool[tier][int(rng.integers(len(pool[tier])))]
                loss,terms,stats=b_or_c(model,item,rng,generator)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{stage} {step} {tier} nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.heads.parameters(),5,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for k,p in model.heads.state_dict().items():
                    ema[k].lerp_(p,1-CONFIG['ema_decay'])
            curves.append(dict(step=step,tier=tier,loss=float(loss.detach()),grad_norm=float(norm),
                               terms={k:float(v.detach()) for k,v in terms.items()},**stats))
            counts[tier]+=1
            uses.update(terms.keys())
            if step==1 or step%100==0:
                status=dict(step=step,stage=stage,counts=counts,loss=curves[-1]['loss'],
                            seconds=elapsed+time.monotonic()-start,
                            peak_MiB=torch.cuda.max_memory_allocated()/2**20)
                save_json(dest/'progress.json',status)
                print('STAGE2',status,flush=True)
            if step%2000==0 or step==steps:
                raw=output_state(model.heads)
                checks={'raw':monitor(model)}
                model.heads.load_state_dict(ema)
                checks['EMA']=monitor(model)
                model.heads.load_state_dict(raw)
                save_json(dest/f'step-{step:06d}-monitor.json',checks)
                save_json(dest/f'step-{step:06d}-curve.json',curves)
                path=dest/f'step-{step:06d}.pt'
                tmp=path.with_suffix('.part')
                torch.save(dict(step=step,model=raw,ema={k:v.cpu() for k,v in ema.items()},
                           rng=rng.bit_generator.state,pixel_rng=generator.get_state(),optimizer=opt.state_dict(),
                           seconds=elapsed+time.monotonic()-start),tmp)
                tmp.replace(path)
        if start_step<=steps:
            save_json(dest/'training.json',dict(steps=steps,counts=counts,uses=uses,
                sampled={k:v/steps for k,v in counts.items()},seconds=elapsed+time.monotonic()-start,
                peak_MiB=torch.cuda.max_memory_allocated()/2**20,loss_weights=WEIGHTS,
                full_resolution_monitor=True,reference_steps=2000))
        # Retain the optimizer and raw endpoint when continuing the second stage.
    audit=read(OUT/'S1.json')
    item=tuple(audit['identity_frame'])
    x,_=sampled(item)
    geom=model.geometry(x)
    reference=data.load(OUT/'S1-frozen-reference.npz')
    for k,v in reference.items():
        np.testing.assert_array_equal(v,geom[k].cpu().numpy())
    save_json(OUT/'S1-frozen-after.json',dict(passed=True,scene=item[1],frame=data.TRAIN_FRAMES[item[2]],outputs=list(reference)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('job',choices=('check','train'))
    args=p.parse_args()
    if args.job=='train':
        train()
    else:
        guard(gpu=True)
        smoke(Model(),read(OUT/'A/manifest.json'))
