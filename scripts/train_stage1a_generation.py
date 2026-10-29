"""Four fixed-budget generation probes and paired held-out evaluation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.bokeh_generation_probe import VideoGenerationProbe
from src.focus_relative_probe import read_json,save_json,sha
from scripts.prepare_stage1a_generation import C,CFG,OUT,PREFIX,boxes,setup


def state_digest(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_csv(path,rows):
    assert not Path(path).exists()
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def focus_masks():
    masks=np.zeros((2,1,252,448),np.float32)
    for i,(x,y,w,h) in enumerate(boxes()):masks[i,:,y:y+h,x:x+w]=1
    return torch.from_numpy(masks).cuda()


def depth_conditions(depth):
    log=np.log(depth);center=np.median(log,axis=(-2,-1),keepdims=True)
    result=[]
    for x,y,w,h in boxes():
        ref=np.log(np.median(depth[:,y:y+h,x:x+w],axis=(-2,-1)))[:,None,None]
        result.append(np.stack([log-center,ref-log],1).clip(-4,4)/4)
    return np.stack(result,1).astype(np.float16)


def load_cache(scenes,kind):
    rgb=[];targets=[];conds=[]
    for s in scenes:
        d=OUT/'data'/s['id'];rgb.append(torch.from_numpy(np.load(d/'rgb.npy')))
        targets.append(torch.from_numpy(np.load(d/'targets.npy')))
        if kind=='Tokens':conds.append(torch.from_numpy(np.load(d/'features/tokens.npy')))
        elif kind in ['Depth','Oracle']:
            z=np.load(d/('GT_depth.npy' if kind=='Oracle' else 'features/depth.npy'))
            conds.append(torch.from_numpy(depth_conditions(z)))
    return dict(rgb=torch.cat(rgb).cuda(),target=torch.cat(targets).cuda(),condition=torch.cat(conds).cuda() if conds else None)


def batch(cache,q,kind,masks):
    q=torch.as_tensor(q,device='cuda',dtype=torch.long)
    ids=q[:,0,None]*25+q[:,1,None]*5+torch.arange(5,device='cuda')[None]
    f,k=q[:,2],q[:,3]
    rgb=cache['rgb'][ids].float()/255
    target=cache['target'][ids,f[:,None],k[:,None]].float()
    cond=None
    if kind=='Tokens':cond=cache['condition'][ids]
    elif kind!='RGB':cond=cache['condition'][ids,f[:,None]]
    return rgb,masks[f],torch.tensor(C['data']['K'],device='cuda')[k],cond,target


def train(kind):
    setup(True);torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed'])
    d=OUT/'heads'/kind;d.mkdir(parents=True,exist_ok=True)
    if (d/'training.json').exists():print('REUSE_TRAIN',kind);return
    t0=time.monotonic();cache=load_cache([s for s in C['scenes'] if s['split']=='train'],kind);masks=focus_masks()
    model=VideoGenerationProbe(kind,C['seed']).cuda()
    trunk=state_digest({k:v for k,v in model.state_dict().items() if not k.startswith(('proj.','interaction.'))})
    opt=torch.optim.AdamW(model.parameters(),lr=C['train']['lr'],weight_decay=C['train']['weight_decay'])
    order=np.load(OUT/'schedule.npy');questions=np.load(OUT/'questions.npy')
    assert order.size==4000 and questions.shape==(160,4)
    curves=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t1=time.monotonic()
    for step,indices in enumerate(order.reshape(2000,2),1):
        rgb,mask,k,cond,target=batch(cache,questions[indices],kind,masks)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(rgb,mask,k,cond)
        loss=F.l1_loss(pred.float(),target);assert torch.isfinite(loss),step
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(gn),step
        opt.step()
        if step==1 or step%100==0:
            curves.append(dict(step=step,L1=float(loss.detach()),gradient_norm=float(gn)))
            print('TRAIN',kind,step,float(loss.detach()),'elapsed',round(time.monotonic()-t1,1),flush=True)
    torch.cuda.synchronize();seconds=time.monotonic()-t1
    cp=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=2000,kind=kind,config_sha256=sha(CFG))
    assert not (d/'final.pt').exists();torch.save(cp,d/'final.pt')
    record=dict(kind=kind,steps=2000,clip_exposures=4000,frame_exposures=20000,
        initial_trunk_sha256=trunk,schedule_sha256=sha(OUT/'schedule.npy'),config_sha256=sha(CFG),
        parameters=sum(p.numel() for p in model.parameters()),adapter_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith(('proj.','interaction.'))),
        training_seconds=seconds,load_train_save_seconds=time.monotonic()-t0,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,curves=curves,checkpoint_sha256=sha(d/'final.pt'),
        no_development_checkpoint_selection=True,no_pretrained_backbone_training=True)
    save_json(d/'training.json',record);save_json(str(PREFIX)+f'-train-{kind}.json',record)
    print('COMPLETE',kind,seconds,flush=True)


def model_load(kind):
    cp=torch.load(OUT/'heads'/kind/'final.pt',map_location='cuda',weights_only=True)
    assert cp['step']==2000 and cp['config_sha256']==sha(CFG)
    model=VideoGenerationProbe(kind,C['seed']).cuda().eval();model.load_state_dict(cp['model'],strict=True)
    return model


def region_masks(alpha):
    out=[]
    kernel=np.ones((5,5),np.uint8)
    for a in alpha:
        valid=np.ones(a.shape,bool);valid[:12]=False;valid[-12:]=False;valid[:,:12]=False;valid[:,-12:]=False
        outer=cv2.dilate((a>.01).astype(np.uint8),kernel)>0
        inner=cv2.erode((a>.99).astype(np.uint8),kernel)>0
        bg=cv2.erode((a<.01).astype(np.uint8),kernel)>0
        out.append(dict(all=valid,boundary=valid&outer&~inner,foreground_core=valid&inner,background_core=valid&bg))
    return out


def score_video(pred,target,regions,full=True):
    from skimage.metrics import structural_similarity
    rows=[]
    for t,(p,y) in enumerate(zip(pred,target)):
        diff=p.astype(np.float32)-y.astype(np.float32)
        for region,mask in regions[t].items():
            err=diff[:,mask];mse=float(np.mean(err**2)) if err.size else None
            row=dict(frame=t,region=region,pixels=int(mask.sum()),MAE=float(np.abs(err).mean()) if err.size else None,
                PSNR=float(-10*np.log10(max(mse,1e-12))) if mse is not None else None)
            if region=='all' and full:
                row['SSIM']=float(structural_similarity(y[:,12:-12,12:-12].transpose(1,2,0).astype(np.float32),p[:,12:-12,12:-12].transpose(1,2,0).astype(np.float32),data_range=1.,channel_axis=2))
            rows.append(row)
    return rows


def evaluate():
    setup(True)
    if (OUT/'evaluation.json').exists():print('REUSE_EVALUATION');return
    # No evaluation until all four fixed endpoints exist.
    records=[read_json(OUT/'heads'/k/'training.json') for k in C['groups']]
    assert len({r['initial_trunk_sha256'] for r in records})==1
    assert len({r['schedule_sha256'] for r in records})==1
    masks=focus_masks();rows=[];checks=[]
    for s in [s for s in C['scenes'] if s['split']!='train']:
        dest=OUT/'predictions'/s['id'];dest.mkdir(parents=True,exist_ok=True)
        data=OUT/'data'/s['id'];targets=np.load(data/'targets.npy');rgb=np.load(data/'rgb.npy').astype(np.float32)/255
        regions=region_masks(np.load(data/'alpha.npy').astype(np.float32))
        for fi in range(2):
            for ki,K in enumerate(C['data']['K']):
                for r in score_video(rgb,targets[:,fi,ki],regions):rows.append(dict(scene=s['id'],split=s['split'],kind='Identity',focus=fi,K=K,**r))
        for kind in C['groups']:
            model=model_load(kind);cache=load_cache([s],kind)
            for fi in range(2):
                for ki,K in enumerate(C['data']['K']):
                    preds=[];ab=[]
                    with torch.inference_mode():
                        for g in range(5):
                            x,mask,k,cond,target=batch(cache,np.array([[0,g,fi,ki]]),kind,masks)
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                p=model(x,mask,k,cond)
                                zero=model(x,mask,torch.zeros_like(k),cond)
                                assert torch.equal(zero,x)
                                if kind!='RGB':ab.append(model(x,mask,k,cond,zero_condition=True)[0].cpu().numpy())
                            assert torch.isfinite(p).all();preds.append(p[0].cpu().numpy())
                    pred=np.concatenate(preds);np.save(dest/f'{kind}-f{fi}-k{int(K)}.npy',pred.astype(np.float16))
                    # Metrics use the stored float16 product consistently.
                    pred=pred.astype(np.float16)
                    for r in score_video(pred,targets[:,fi,ki],regions):rows.append(dict(scene=s['id'],split=s['split'],kind=kind,focus=fi,K=K,**r))
                    if ab:
                        a=np.concatenate(ab).astype(np.float16)
                        np.save(dest/f'{kind}-zero-f{fi}-k{int(K)}.npy',a)
                        for r in score_video(a,targets[:,fi,ki],regions,False):rows.append(dict(scene=s['id'],split=s['split'],kind=kind+'_zero_condition',focus=fi,K=K,**r))
            # Fresh model reload with matched five-frame autocast reproduces last group.
            clone=model_load(kind)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):q=clone(x,mask,k,cond)[0].cpu().numpy().astype(np.float16)
            assert np.array_equal(q,pred[-5:])
            checks.append(dict(scene=s['id'],kind=kind,zero_K_identity=True,last_group_reloaded_bitwise=True))
            del clone,model,cache
            print('EVALUATED',s['id'],kind,flush=True)
    source=[]
    for s in [s for s in C['scenes'] if s['split']!='train']:
        for kind in sorted({r['kind'] for r in rows}):
            for reg in ['all','boundary','foreground_core','background_core']:
                rr=[r for r in rows if r['scene']==s['id'] and r['kind']==kind and r['region']==reg]
                record=dict(scene=s['id'],split=s['split'],kind=kind,region=reg)
                for metric in ['PSNR','MAE','SSIM']:
                    vals=[r[metric] for r in rr if r.get(metric) is not None]
                    record[metric]=float(np.mean(vals)) if vals else None
                source.append(record)
    summary=[]
    for split in ['dev','test']:
        for kind in sorted({r['kind'] for r in rows}):
            for reg in ['all','boundary','foreground_core','background_core']:
                rr=[r for r in source if (r['split'],r['kind'],r['region'])==(split,kind,reg)]
                record=dict(split=split,kind=kind,region=reg,sources=len(rr))
                for metric in ['PSNR','MAE','SSIM']:
                    vals=[r[metric] for r in rr if r[metric] is not None];record[metric]=float(np.mean(vals)) if vals else None
                summary.append(record)
    def get(split,kind,reg,metric):return next(r[metric] for r in summary if (r['split'],r['kind'],r['region'])==(split,kind,reg))
    decisions={}
    for split in ['dev','test']:
        delta=get(split,'Tokens','all','PSNR')-get(split,'Depth','all','PSNR')
        boundary=get(split,'Tokens','boundary','MAE')/get(split,'Depth','boundary','MAE')-1
        oracle=get(split,'Oracle','all','PSNR')-get(split,'Identity','all','PSNR')
        decisions[split]=dict(Tokens_minus_Depth_PSNR=delta,Tokens_boundary_MAE_relative_increase=boundary,
            Oracle_minus_Identity_PSNR=oracle,tokens_screen_passed=bool(delta>=.3 and boundary<=.02),oracle_sanity=bool(oracle>=1.))
    rec=dict(config_sha256=sha(CFG),source=source,summary=summary,decisions=decisions,checks=checks,
        training=records,not_real_Bokeh_benchmark=True,visual_review='user pending')
    save_json(OUT/'evaluation.json',rec);save_json(str(PREFIX)+'-results.json',rec)
    write_csv(str(PREFIX)+'-frames.csv',rows);write_csv(str(PREFIX)+'-per-source.csv',source);write_csv(str(PREFIX)+'-summary.csv',summary)
    print('DECISIONS',decisions,flush=True)


def report():
    import imageio_ffmpeg
    dest=OUT/'report';dest.mkdir(exist_ok=True);html='<meta charset="utf-8"><h1>Stage1a 合成生成对照</h1><p>输入 / 合成渲染参考 / RGB组 / 最终深度 / 中间特征 / Oracle。Oracle使用真实合成深度，非可部署方法。固定焦点0、K32，25帧，非真实视频画质结论。</p>'
    for s in [s for s in C['scenes'] if s['split']!='train']:
        d=OUT/'data'/s['id'];p=OUT/'predictions'/s['id']
        arrays=[np.load(d/'rgb.npy').astype(np.float32)/255,np.load(d/'targets.npy')[:,0,1]]
        arrays += [np.load(p/f'{k}-f0-k32.npy') for k in C['groups']]
        labels=['Input RGB','Synthetic reference','RGB only','VGGT depth','VGGT tokens','Oracle geometry']
        frames=[];sheet=Image.new('RGB',(1344,564*3),'white')
        for t in range(25):
            canvas=Image.new('RGB',(1344,564),'white');dr=ImageDraw.Draw(canvas)
            for j,(a,l) in enumerate(zip(arrays,labels)):
                xx=j%3*448;yy=j//3*282
                im=Image.fromarray(np.uint8(np.round(np.clip(a[t].transpose(1,2,0),0,1)*255)))
                canvas.paste(im,(xx,yy+30));dr.text((xx+5,yy+8),f'{s["id"]} f{t} {l}',fill='black')
            frames.append(np.asarray(canvas))
            if t in [0,12,24]:sheet.paste(canvas,(0,[0,12,24].index(t)*564))
        sheet.save(dest/f'{s["id"]}.png')
        writer=imageio_ffmpeg.write_frames(str(dest/f'{s["id"]}.mp4'),(1344,564),fps=20,codec='libx264',pix_fmt_in='rgb24',output_params=['-crf','16','-movflags','+faststart'],macro_block_size=1)
        writer.send(None)
        for f in frames:writer.send(f)
        writer.close()
        html+=f'<h2>{s["id"]} / {s["split"]}</h2><video controls loop style="width:100%" src="{s["id"]}.mp4"></video><p>循环24→0不是连续过渡。0.5倍速按钮仅改变播放速度。</p><button onclick="this.previousElementSibling.previousElementSibling.playbackRate=.5">0.5×</button><br><a href="{s["id"]}.png">帧0/12/24原尺寸接触图</a>'
    (dest/'index.html').write_text(html)
    print('REPORT_COMPLETE',dest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','evaluate','report']);p.add_argument('--kind',choices=C['groups']);a=p.parse_args()
    if a.phase=='train':assert a.kind;train(a.kind)
    elif a.phase=='evaluate':evaluate()
    else:report()
