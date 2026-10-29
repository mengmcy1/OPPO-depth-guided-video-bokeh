"""New source-isolated planar video pairs; synthetic renderer references only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.focus_relative_probe import read_json,save_json,sha
from src.synthetic.common import srgb_to_linear,linear_to_srgb
C=read_json(ROOT/'configs/v2-stage1a-generation-20261029.json');OUT=ROOT/C['output']
PREFIX=ROOT/C['record_prefix'];CFG=ROOT/'configs/v2-stage1a-generation-20261029.json'


def boxes():
    w,h=C['data']['width'],C['data']['height'];size=C['data']['focus_box']
    return [(round(x*w-size/2),round(y*h-size/2),size,size) for x,y in C['data']['focus_centers']]


def setup(gpu=False):
    cv2.setNumThreads(1);torch.set_num_threads(6)
    if gpu:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def read_clip(path,alpha=False,start=0):
    cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path
    if start:cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    frames=[]
    for t in range(25):
        ok,bgr=cap.read();assert ok,(path,t)
        # Decode frame, resize both roles to the same bounded grid.
        scale=min(1.,1536/max(bgr.shape[:2]));size=(round(bgr.shape[1]*scale),round(bgr.shape[0]*scale))
        if scale<1:bgr=cv2.resize(bgr,size,interpolation=cv2.INTER_AREA)
        frames.append(bgr[:,:,0] if alpha else cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release();return frames


def prepare():
    setup();OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    if (OUT/'prepared.json').exists():print('REUSE_PREPARED');return
    from scripts.check_benchmark_source_isolation import collect_development
    old={r['source_id'] for r in collect_development()[0]}
    ids=[];rows=[];w,h=C['data']['width'],C['data']['height'];pad=C['data']['padding']
    for s in C['scenes']:
        for k in ['foreground_source_id','background_source_id']:
            assert s[k] not in old and s[k] not in ids,s[k]
            ids.append(s[k])
        dest=OUT/'data'/s['id'];dest.mkdir(exist_ok=True)
        if (dest/'prepared.json').exists():rows.append(read_json(dest/'prepared.json'));continue
        colors=read_clip(ROOT/s['foreground'],start=s.get('source_start_frame',0));alphas=read_clip(ROOT/s['alpha'],True,start=s.get('source_start_frame',0))
        ys=[];xs=[]
        for a in alphas:
            yy,xx=np.where(a>128);assert len(xx)>0,s['id'];ys.extend([yy.min(),yy.max()]);xs.extend([xx.min(),xx.max()])
        x0,x1,y0,y1=map(int,(min(xs),max(xs)+1,min(ys),max(ys)+1))
        scale=min(w*.65/(x1-x0),h*.88/(y1-y0));fw,fh=round((x1-x0)*scale),round((y1-y0)*scale)
        back=np.asarray(Image.open(ROOT/s['background']).convert('RGB').resize((w,h),Image.Resampling.LANCZOS),np.float32)/255
        rgb=[];depth=[];matte=[];foreground=[];focus=[];coverage=[]
        (dest/'rgb').mkdir(exist_ok=True)
        for t,(col,a) in enumerate(zip(colors,alphas)):
            cut=cv2.resize(col[y0:y1,x0:x1],(fw,fh),interpolation=cv2.INTER_AREA).astype(np.float32)/255
            am=cv2.resize(a[y0:y1,x0:x1],(fw,fh),interpolation=cv2.INTER_AREA).astype(np.float32)/255
            px=round(w*.5-fw/2+5*np.sin(2*np.pi*t/25));py=round(h*.53-fh/2)
            assert px>=0 and py>=0 and px+fw<=w and py+fh<=h
            fg=np.zeros((h,w,3),np.float32);alpha=np.zeros((h,w),np.float32)
            fg[py:py+fh,px:px+fw]=cut;alpha[py:py+fh,px:px+fw]=am
            comp=linear_to_srgb(srgb_to_linear(fg)*alpha[...,None]+srgb_to_linear(back)*(1-alpha[...,None]))
            quant=np.uint8(np.round(np.clip(comp,0,1)*255));Image.fromarray(quant).save(dest/'rgb'/f'{t:05d}.png')
            z=np.where(alpha>.5,2.,6.).astype(np.float32)
            foc=[float(np.median(z[y:y+bh,x:x+bw])) for x,y,bw,bh in boxes()]
            focus.append(foc);coverage.append([float(alpha[y:y+bh,x:x+bw].mean()) for x,y,bw,bh in boxes()])
            rgb.append(quant.transpose(2,0,1));depth.append(z);matte.append(alpha);foreground.append(np.uint8(np.round(fg*255)))
        np.save(dest/'rgb.npy',np.stack(rgb));np.save(dest/'GT_depth.npy',np.stack(depth));np.save(dest/'alpha.npy',np.stack(matte).astype(np.float16))
        np.save(dest/'foreground.npy',np.stack(foreground));np.save(dest/'background.npy',np.uint8(np.round(back*255)))
        row=dict(**s,frames=25,union_bbox=[x0,y0,x1,y1],resized_fg_wh=[fw,fh],focus_boxes=boxes(),focus_depth_m=focus,
            focus_alpha_mean=coverage,original_source_hashes={k:sha(ROOT/s[k]) for k in ['foreground','alpha','background']},
            renderer_reference='two known planes, full hidden background, synthetic only')
        save_json(dest/'prepared.json',row);rows.append(row);print('PREPARED',s['id'],flush=True)
    rng=np.random.default_rng(C['seed']);questions=np.array([(s,group,f,k) for s in range(8) for group in range(5) for f in range(2) for k in range(2)],np.int32)
    order=[]
    while len(order)<C['train']['steps']*C['train']['batch_clips']:order.extend(rng.permutation(len(questions)).tolist())
    np.save(OUT/'questions.npy',questions);np.save(OUT/'schedule.npy',np.array(order[:4000],np.int32))
    record=dict(config_sha256=sha(CFG),scenes=rows,train_sources=8,dev_sources=2,test_sources=2,source_overlap=0,
        old_development_overlap=0,questions_per_source=100,training_clip_questions=160,training_exposures_clips=4000,
        schedule_sha256=sha(OUT/'schedule.npy'),same_subject_identity_is_not_audited=True)
    save_json(OUT/'prepared.json',record);save_json(str(PREFIX)+'-data.json',record)


def render():
    setup(True)
    from src.synthetic.public_ray_renderer import load_renderer,render as ray
    module=load_renderer('MPIB');pad=C['data']['padding'];records=[]
    def ten(x):return torch.from_numpy(x).cuda().float().contiguous()
    if not (OUT/'renderer-check.json').exists():
        images=torch.tensor([.2,.4,.6],device='cuda').reshape(1,1,3,1,1).expand(1,1,3,48,64).contiguous()
        alpha=torch.ones((1,1,1,48,64),device='cuda');out=ray(module,images,alpha,[[6.]],32.,2.,51)
        err=float((out-images[:,0]).abs().max());assert err<.001,err
        save_json(OUT/'renderer-check.json',dict(constant_defocus_max_error=err,kernel_sha256=sha(ROOT.parent/'MPIB/bokeh_generator.py'),linear_light=True))
    for s in C['scenes']:
        dest=OUT/'data'/s['id']
        if (dest/'render.json').exists():records.append(read_json(dest/'render.json'));print('REUSE_RENDER',s['id'],flush=True);continue
        meta=read_json(dest/'prepared.json');fg=np.load(dest/'foreground.npy').astype(np.float32)/255
        back=np.load(dest/'background.npy').astype(np.float32)/255;aa=np.load(dest/'alpha.npy').astype(np.float32)
        # Render all25 frames in the same fixed two-layer geometry.
        bg=np.broadcast_to(back,fg.shape)
        images=np.stack([srgb_to_linear(fg),srgb_to_linear(bg)],axis=1).transpose(0,1,4,2,3)
        alpha=np.stack([aa,np.ones_like(aa)],axis=1)[:,:,None]
        images=np.pad(images,((0,0),(0,0),(0,0),(pad,pad),(pad,pad)),mode='edge')
        alpha=np.pad(alpha,((0,0),(0,0),(0,0),(pad,pad),(pad,pad)),mode='edge')
        x,a=ten(images),ten(alpha);z=torch.tensor([2.,6.],device='cuda')[None].expand(25,-1)
        target=np.empty((25,2,2,3,C['data']['height'],C['data']['width']),np.float16)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();checks={}
        with torch.inference_mode():
            for fi in range(2):
                focus=torch.tensor(np.array(meta['focus_depth_m'])[:,fi],device='cuda',dtype=torch.float32)
                for ki,K in enumerate(C['data']['K']):
                    v=ray(module,x,a,z,K,focus,C['data']['samples_per_side'])[:,:,pad:-pad,pad:-pad]
                    linear=v.cpu().numpy();encoded=np.clip(linear_to_srgb(linear),0,1)
                    assert np.isfinite(encoded).all();target[:,fi,ki]=encoded.astype(np.float16)
                    if s['id']=='gen_00' and fi==0 and ki==1:
                        hi=ray(module,x[:1],a[:1],z[:1],K,focus[:1],101)[:,:,pad:-pad,pad:-pad].cpu().numpy()
                        high=np.clip(linear_to_srgb(hi),0,1);diff=encoded[:1]-high
                        checks['sample51_vs101']=dict(MAE=float(np.abs(diff).mean()),PSNR=float(-10*np.log10(np.mean(diff**2)+1e-15)),max_abs=float(np.abs(diff).max()))
            zero=ray(module,x[:1],a[:1],z[:1],0.,2.,51)[:,:,pad:-pad,pad:-pad].cpu().numpy()
            rgb=np.load(dest/'rgb.npy')[:1].astype(np.float32)/255
            ze=np.clip(linear_to_srgb(zero),0,1)-rgb
            checks['zero_K_vs_quantized_RGB']=dict(MAE=float(np.abs(ze).mean()),max_abs=float(np.abs(ze).max()))
            assert np.abs(ze).mean()<.003,checks
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        assert not (dest/'targets.npy').exists();np.save(dest/'targets.npy',target)
        row=dict(scene=s['id'],target_shape=list(target.shape),elapsed_seconds=elapsed,checks=checks,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,target_sha256=sha(dest/'targets.npy'))
        save_json(dest/'render.json',row);records.append(row);print('RENDERED',s['id'],round(elapsed,2),checks,flush=True)
        del x,a,z,v
    if not (OUT/'rendered.json').exists():
        record=dict(config_sha256=sha(CFG),scenes=records,targets=1200,not_real_captured_GT=True)
        save_json(OUT/'rendered.json',record);save_json(str(PREFIX)+'-renderer.json',record)


def features():
    setup(True)
    from scripts.run_v2_stage0 import imports
    base=read_json(ROOT/C['geometry']['base_stage0_config']);VGGT,_,loader=imports(base)
    model=VGGT(enable_camera=False,enable_point=False,enable_track=False)
    path=ROOT/base['models']['VGGT']['weight'];assert sha(path)==base['models']['VGGT']['sha256']
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict({k:v for k,v in state.items() if k.startswith(('aggregator.','depth_head.'))},strict=True,assign=True);del state
    model.eval().requires_grad_(False).cuda();records=[]
    for s in C['scenes']:
        dest=OUT/'data'/s['id'];fd=dest/'features';fd.mkdir(exist_ok=True)
        if (fd/'complete.json').exists():records.append(read_json(fd/'complete.json'));continue
        depths=[];feats=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        for g in range(5):
            x=loader([str(dest/'rgb'/f'{t:05d}.png') for t in range(g*5,g*5+5)],mode='crop').cuda()
            assert x.shape==(5,3,294,518),x.shape
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                tokens,start_idx=model.aggregator(x[None]);assert start_idx==5
                f=tokens[17][0,:,start_idx:].reshape(5,21,37,2048).permute(0,3,1,2)
                with torch.autocast('cuda',enabled=False):d,_=model.depth_head(tokens,images=x[None],patch_start_idx=start_idx)
                d=F.interpolate(d[0].permute(0,3,1,2),size=(252,448),mode='bilinear',align_corners=False)
            assert torch.isfinite(f).all() and torch.isfinite(d).all() and (d>0).all()
            feats.append(f.float().cpu().numpy().astype(np.float16));depths.append(d[:,0].float().cpu().numpy());del tokens,x,d,f
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        np.save(fd/'tokens.npy',np.concatenate(feats));np.save(fd/'depth.npy',np.concatenate(depths))
        rec=dict(scene=s['id'],seconds=elapsed,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            input_grid=[294,518],feature_grid=[21,37],output_grid=[252,448],same_forward_for_depth_tokens=True,
            model_sha256=base['models']['VGGT']['sha256'])
        save_json(fd/'complete.json',rec);records.append(rec);print('FEATURES',s['id'],round(elapsed,2),flush=True)
    if not (OUT/'features.json').exists():save_json(OUT/'features.json',dict(scenes=records,backbone_frozen=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','render','features']);a=p.parse_args()
    {'prepare':prepare,'render':render,'features':features}[a.phase]()
