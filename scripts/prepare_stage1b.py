"""Input-determined blur reference and independent frozen DINO feature cache."""
import argparse
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.focus_relative_probe import read_json,save_json,sha
from src.synthetic.common import srgb_to_linear,linear_to_srgb
from scripts import prepare_stage1a_generation as old
C=read_json(ROOT/'configs/v2-stage1b-20261029.json');OUT=ROOT/C['output'];PREFIX=ROOT/C['record_prefix']


def signed_base_coc(depth):
    values=[]
    for x,y,w,h in old.boxes():
        ref=np.median(depth[:,y:y+h,x:x+w],axis=(-1,-2))[:,None,None]
        values.append(1/depth-1/ref)
    return np.stack(values,axis=1).astype(np.float32)


def blur_bank(rgb):
    linear=srgb_to_linear(rgb.astype(np.float32))
    bank=[linear]
    for r in range(1,12):
        sigma=r/2;size=2*math.ceil(3*sigma)+1
        bank.append(cv2.GaussianBlur(linear,(size,size),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REFLECT_101))
    return np.stack(bank)


def observable_target(bank,signed_coc):
    r=np.abs(signed_coc);assert r.max()<=11 and np.isfinite(r).all()
    lo=np.floor(r).astype(np.int32);hi=np.minimum(lo+1,11);a=(r-lo)[...,None]
    yy,xx=np.indices(r.shape)
    linear=bank[lo,yy,xx]*(1-a)+bank[hi,yy,xx]*a
    return linear_to_srgb(linear).clip(0,1).astype(np.float32)


def prepare():
    old.setup();OUT.mkdir(exist_ok=True,parents=True);(OUT/'data').mkdir(exist_ok=True)
    if (OUT/'targets.json').exists():print('REUSE_TARGETS');return
    rng=np.random.default_rng(20261029);a=rng.random((32,40,3),dtype=np.float32)
    zero=observable_target(blur_bank(a),np.zeros((32,40),np.float32));assert np.max(abs(zero-a))<3e-7
    const=np.full((32,40,3),.4,np.float32);b=blur_bank(const);r=rng.random((32,40),dtype=np.float32)*10
    uniform=observable_target(b,r);assert np.max(abs(uniform-const))<3e-7
    assert np.array_equal(observable_target(b,r),observable_target(b,-r))
    controls=dict(zero_CoC_max_error=float(abs(zero-a).max()),constant_max_error=float(abs(uniform-const).max()),signed_magnitude_symmetry=True,
        target_function_inputs='RGB and signed CoC only; no alpha/hidden layers')
    records=[]
    for s in old.C['scenes']:
        dest=OUT/'data'/s['id'];dest.mkdir(exist_ok=True)
        if (dest/'targets.json').exists():records.append(read_json(dest/'targets.json'));continue
        source=old.OUT/'data'/s['id'];rgb=np.load(source/'rgb.npy').transpose(0,2,3,1).astype(np.float32)/255
        depth=np.load(source/'GT_depth.npy');coc=signed_base_coc(depth)
        target=np.empty((25,2,2,3,252,448),np.float16);start=time.monotonic()
        for t in range(25):
            bank=blur_bank(rgb[t])
            for f in range(2):
                for ki,K in enumerate(old.C['data']['K']):target[t,f,ki]=observable_target(bank,K*coc[t,f]).transpose(2,0,1).astype(np.float16)
        assert np.isfinite(target).all();np.save(dest/'observable.npy',target)
        rec=dict(scene=s['id'],shape=list(target.shape),seconds=time.monotonic()-start,target_sha256=sha(dest/'observable.npy'),
            source_RGB_sha256=sha(source/'rgb.npy'),source_depth_sha256=sha(source/'GT_depth.npy'))
        save_json(dest/'targets.json',rec);records.append(rec);print('OBSERVABLE_TARGET',s['id'],round(rec['seconds'],2),flush=True)
    rec=dict(controls=controls,scenes=records,config_sha256=sha(ROOT/'configs/v2-stage1b-20261029.json'),targets=1200,
        original_data_immutable=True,not_physical_bokeh_reference=True)
    save_json(OUT/'targets.json',rec);save_json(str(PREFIX)+'-targets.json',rec)


def dino():
    old.setup(True)
    if (OUT/'DINO.json').exists():print('REUSE_DINO');return
    from scripts.run_v2_stage0 import imports
    config=read_json(ROOT/'configs/v2-stage0-20261029.json');_,vit_large,loader=imports(config)
    model=vit_large(img_size=518,patch_size=14,num_register_tokens=4,interpolate_antialias=True,interpolate_offset=0.,block_chunks=0,init_values=1.)
    weight=ROOT/config['models']['P0']['weight']
    previous=read_json(ROOT/config['output']/'P0-extraction.json');assert sha(weight)==previous['weight_sha256']
    state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(state,strict=True,assign=True);del state
    model.eval().requires_grad_(False).cuda();records=[]
    for s in old.C['scenes']:
        dest=OUT/'data'/s['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'DINO.json').exists():records.append(read_json(dest/'DINO.json'));continue
        features=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        for g in range(5):
            x=loader([str(old.OUT/'data'/s['id']/'rgb'/f'{t:05d}.png') for t in range(g*5,g*5+5)],mode='crop').cuda()
            assert x.shape==(5,3,294,518)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
                f=model.forward_features((x-mean)/std)['x_norm_patchtokens'];assert f.shape==(5,777,1024)
                f=torch.cat([f,f.mean(0,keepdim=True).expand_as(f)],-1).reshape(5,21,37,2048).permute(0,3,1,2)
            assert torch.isfinite(f).all();features.append(f.float().cpu().numpy().astype(np.float16));del x,f
        torch.cuda.synchronize();np.save(dest/'DINO.npy',np.concatenate(features))
        rec=dict(scene=s['id'],seconds=time.monotonic()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            output_shape=[25,2048,21,37],weight_sha256=previous['weight_sha256'],frozen=True)
        save_json(dest/'DINO.json',rec);records.append(rec);print('DINO',s['id'],flush=True)
    rec=dict(scenes=records,weight_sha256=previous['weight_sha256'],parameter_count=sum(p.numel() for p in model.parameters()),
        compared_to_VGGT='same RGB five-frame groups/grid/adapter; feature stages and cross-frame processing differ')
    save_json(OUT/'DINO.json',rec);save_json(str(PREFIX)+'-DINO.json',rec)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','dino']);a=p.parse_args();{'prepare':prepare,'dino':dino}[a.phase]()
