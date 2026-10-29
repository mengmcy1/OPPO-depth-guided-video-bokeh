"""Executable five-way frozen-feature Stage0, with reusable completed items.

Run with any2bokeh Python, CUDA_VISIBLE_DEVICES=0. Large feature caches,
checkpoints and prediction arrays remain under the ignored output root.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.focus_relative_probe import (read_json,save_json,sha,build_labels,focus_boxes,make_focus_masks,
    confusion,metrics,rule_score,threshold_calibration,RelationProbe,self_checks)
CFG=ROOT/'configs/v2-stage0-20261029.json'


def setup(c,gpu=False):
    torch.set_num_threads(6);cv2.setNumThreads(1)
    torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    if gpu:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.manual_seed_all(c['seed'])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def write_csv(path,rows):
    path=Path(path);assert not path.exists()
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,keys,lineterminator='\n');writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def imports(c):
    repo=(ROOT/c['models']['VGGT']['repo']).resolve()
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==c['models']['VGGT']['commit']
    sys.path.insert(0,str(repo))
    from vggt.models.vggt import VGGT
    from vggt.layers.vision_transformer import vit_large
    from vggt.utils.load_fn import load_and_preprocess_images
    return VGGT,vit_large,load_and_preprocess_images


def prepare(c,out):
    if (out/'prepared.json').exists():
        assert read_json(out/'prepared.json')['config_sha256']==sha(CFG)
        print('REUSE_PREPARED',flush=True);return
    checks=self_checks();manifest=build_labels(ROOT,c,CFG)
    assert manifest['train_questions']==500 and manifest['dev_questions']==250
    # Fixed RNG schedule, reused byte-for-byte by each trainable group.
    schedule=[];rng=np.random.default_rng(c['seed']);need=c['training']['steps']*c['training']['batch_size']
    while len(schedule)<need:schedule.extend(rng.permutation(manifest['train_questions']).tolist())
    np.save(out/'training-schedule.npy',np.asarray(schedule[:need],np.int32))
    save_json(out/'prepared.json',dict(config_sha256=sha(CFG),checks=checks,train_questions=500,dev_questions=250,
              training_schedule_sha256=sha(out/'training-schedule.npy'),training_class_counts=manifest['training_class_counts'],
              training_loss_weights=manifest['training_loss_weights'],frozen_before_dev_results=True))
    if not (out/'implementation-checks.json').exists():
        save_json(out/'implementation-checks.json',dict(self_checks=checks,
            scope='Independent reproduction; detailed original-run recovery is retained in v2-stage0-20261029-verification.json.'))
    write_csv(ROOT/(c['record_prefix']+'-manifest.csv'),manifest['questions'])
    # Fixed input/label previews for every source, before model extraction.
    previews=out/'alignment';previews.mkdir(exist_ok=True)
    palette=np.array([[237,105,65],[100,194,108],[77,128,222],[160,160,160]],np.uint8)
    for scene in c['sequences']:
        name=scene['sequence'];data=np.load(out/'data'/f'{name}.npz')
        sheet=Image.new('RGB',(518*3,254*3),'white');draw=ImageDraw.Draw(sheet)
        for ri,t in enumerate([0,12,24]):
            frame=scene['frames'][t]
            rgb=Image.open(ROOT/c['raw_root']/'training/clean'/name/f'frame_{frame:04d}.png').convert('RGB').resize((518,224),Image.Resampling.BICUBIC)
            dr=ImageDraw.Draw(rgb);x,y,w,h=focus_boxes(c)[1];dr.rectangle([x,y,x+w-1,y+h-1],outline='yellow',width=2)
            lab=data['labels'][t,1];idx=np.where(lab<3,lab,3)
            for j,image in enumerate([rgb,Image.fromarray(palette[idx]),Image.fromarray(np.uint8(data['valid'][t])*255).convert('RGB')]):
                sheet.paste(image,(j*518,ri*254+30));draw.text((j*518+4,ri*254+7),f'{name} frame{frame}: '+['RGB/fixed focus1','GT near/similar/far','numeric validity'][j],fill='black')
        sheet.save(previews/(name+'.png'))
    print('PREPARED',manifest['training_class_counts'],manifest['training_loss_weights'],flush=True)


def extract(c,out,kind):
    if (out/f'{kind}-extraction.json').exists():
        prior=read_json(out/f'{kind}-extraction.json')
        assert len(prior['groups'])==30
        print('REUSE_COMPLETE_EXTRACTION',kind,flush=True);return
    setup(c,True);VGGT,vit_large,load_images=imports(c)
    start=time.monotonic()
    if kind=='VGGT':
        model=VGGT(enable_camera=False,enable_point=False,enable_track=False)
        weight=ROOT/c['models']['VGGT']['weight'];assert sha(weight)==c['models']['VGGT']['sha256']
        state=torch.load(weight,weights_only=True,map_location='cpu',mmap=True)
        full_parameters=sum(v.numel() for v in state.values())
        filtered={k:v for k,v in state.items() if k.startswith('aggregator.') or k.startswith('depth_head.')}
        model.load_state_dict(filtered,strict=True,assign=True);del state,filtered
    else:
        model=vit_large(img_size=518,patch_size=14,num_register_tokens=4,interpolate_antialias=True,interpolate_offset=0.,block_chunks=0,init_values=1.)
        weight=ROOT/c['models']['P0']['weight']
        state=torch.load(weight,weights_only=True,map_location='cpu',mmap=True)
        model.load_state_dict(state,strict=True,assign=True);full_parameters=sum(v.numel() for v in state.values());del state
    model.eval().requires_grad_(False).cuda()
    load_seconds=time.monotonic()-start;records=[]
    assert not any(p.requires_grad for p in model.parameters())
    for scene in c['sequences']:
        name=scene['sequence'];dest=out/'features'/name;dest.mkdir(parents=True,exist_ok=True)
        for gi,frames in enumerate(c['context']['groups_native_frames']):
            stamp=dest/f'{kind}-group{gi}.json'
            if stamp.exists():
                records.append(read_json(stamp));print('REUSE',kind,name,gi,flush=True);continue
            paths=[str(ROOT/c['raw_root']/'training/clean'/name/f'frame_{frame:04d}.png') for frame in frames]
            before=time.monotonic();images=load_images(paths,mode='crop');assert tuple(images.shape)==(5,3,224,518)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                image_cuda=images.cuda()
                if kind=='VGGT':
                    tokens,patch_start=model.aggregator(image_cuda.unsqueeze(0))
                    assert patch_start==5 and len(tokens)==24 and tokens[17].shape==(1,5,597,2048)
                    feature=tokens[c['models']['VGGT']['layer_zero_based']][0,:,patch_start:]
                    with torch.autocast('cuda',enabled=False):
                        depth,confidence=model.depth_head(tokens,images=image_cuda.unsqueeze(0),patch_start_idx=patch_start)
                    assert depth.shape==(1,5,224,518,1)
                    depth=depth[0,...,0]
                    assert torch.isfinite(depth).all() and (depth>0).all()
                else:
                    mean=image_cuda.new_tensor([.485,.456,.406])[None,:,None,None]
                    std=image_cuda.new_tensor([.229,.224,.225])[None,:,None,None]
                    f=model.forward_features((image_cuda-mean)/std)['x_norm_patchtokens']
                    assert f.shape==(5,592,1024)
                    feature=torch.cat([f,f.mean(dim=0,keepdim=True).expand_as(f)],dim=-1)
                feature=feature.reshape(5,16,37,2048).permute(0,3,1,2).contiguous()
                assert torch.isfinite(feature).all()
            torch.cuda.synchronize();compute_seconds=time.monotonic()-t0
            np.save(dest/f'{kind}-group{gi}.npy',feature.float().cpu().numpy().astype(np.float16))
            if kind=='VGGT':np.save(dest/f'depth-group{gi}.npy',depth.float().cpu().numpy())
            row=dict(kind=kind,sequence=name,group_index=gi,frames=frames,input_shape=list(images.shape),feature_shape=list(feature.shape),feature_dtype='float16 CPU cache from BF16 backbone',
                     patch_start=5 if kind=='VGGT' else 'DINO forward_features already removes cls+4 register tokens',compute_seconds=compute_seconds,
                     elapsed_preprocess_forward_cache_seconds=time.monotonic()-before,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,all_finite=True,pretrained_frozen=True)
            save_json(stamp,row);records.append(row)
            if kind=='VGGT' and name==c['sequences'][0]['sequence'] and gi==0:
                save_json(out/'VGGT-interface.json',dict(**row,official_import_file=str(sys.modules['vggt.models.vggt'].__file__),
                     source_commit=c['models']['VGGT']['commit'],spatial_grid=[16,37],special_token_count=5,depth_positive=True,
                     minimal_training_source_group_passed_before_all_features=True))
            print('EXTRACTED',kind,name,gi,round(compute_seconds,3),'s',round(row['peak_allocated_GiB'],3),'GiB',flush=True)
            del feature,image_cuda,images
            if kind=='VGGT':del tokens,depth,confidence
            else:del f,mean,std
        if kind=='VGGT':
            path=dest/'depth.npy'
            if not path.exists():np.save(path,np.concatenate([np.load(dest/f'depth-group{gi}.npy') for gi in range(5)]))
    save_json(out/f'{kind}-extraction.json',dict(kind=kind,weight=str(weight.relative_to(ROOT)),weight_sha256=sha(weight),
         full_checkpoint_tensor_parameters=full_parameters,executed_model_parameters=parameter_count(model),load_seconds=load_seconds,
         groups=records,total_compute_seconds=sum(r['compute_seconds'] for r in records),
         total_preprocess_forward_cache_seconds=sum(r['elapsed_preprocess_forward_cache_seconds'] for r in records),
         peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in records),peak_reserved_GiB=max(r['peak_reserved_GiB'] for r in records),
         python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),backbone_frozen=True))


def scene_features(c,out,name,method):
    if method=='P2plus':
        d=torch.from_numpy(np.load(out/'features'/name/'depth.npy')).unsqueeze(1)
        log=torch.log(d);center=log.flatten(2).median(dim=2).values[:,:,None,None]
        # Historical Stage0 defaults stay unchanged. A separate, frozen audit
        # config may change this one spatial-grid factor for P2+ only.
        grid=tuple(c.get('P2plus_spatial_grid',[16,37]))
        return F.interpolate(log-center,size=grid,mode='nearest-exact')
    kind='P0' if method=='P0' else 'VGGT'
    return torch.from_numpy(np.concatenate([np.load(out/'features'/name/f'{kind}-group{g}.npy') for g in range(5)]))


def state_digest(state):
    h=hashlib.sha256()
    for key,value in state.items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(c,out,method):
    setup(c,True);dest=out/'heads'/method;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'training.json').exists():print('REUSE_TRAIN',method,flush=True);return
    start=time.monotonic();features=[];labels=[];questions=[];offset=0
    for s in c['sequences']:
        if s['split']!='pilot_train':continue
        name=s['sequence'];f=scene_features(c,out,name,method)
        data=np.load(out/'data'/f'{name}.npz');lab=data['labels']
        for t in range(25):
            for fi in range(5):
                if (lab[t,fi]<3).any():questions.append([offset+t,fi])
        features.append(f);labels.append(torch.from_numpy(lab));offset+=25
    f=torch.cat(features).cuda();y=torch.cat(labels).cuda();del features,labels
    q=torch.tensor(questions,device='cuda');masks=torch.from_numpy(make_focus_masks(c)).unsqueeze(1).cuda()
    assert len(questions)==500
    model=RelationProbe(f.shape[1],c).cuda()
    # Identical relation initialization even when scalar projection has fewer parameters.
    torch.manual_seed(c['seed']+1)
    for layer in model.relation.modules():
        if isinstance(layer,torch.nn.Conv2d):layer.reset_parameters()
    initial_relation_sha=state_digest(model.relation.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay'])
    weights=torch.tensor(read_json(out/'prepared.json')['training_loss_weights'],device='cuda',dtype=torch.float32)
    order=np.load(out/'training-schedule.npy');assert order.size==8000
    curves=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();loop_start=time.monotonic()
    for step,idx in enumerate(order.reshape(c['training']['steps'],c['training']['batch_size']),1):
        ids=q[torch.as_tensor(idx.astype(np.int64),device='cuda')];frame,fi=ids[:,0],ids[:,1]
        opt.zero_grad(set_to_none=True)
        pred=model(f[frame],masks[fi]);loss=F.cross_entropy(pred,y[frame,fi].long(),weight=weights,ignore_index=255)
        assert torch.isfinite(loss),step
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm),step
        opt.step()
        if step==1 or step%100==0:
            value=float(loss.detach());curves.append(dict(step=step,loss=value,gradient_norm=float(norm)));print('TRAIN',method,step,value,flush=True)
    torch.cuda.synchronize();seconds=time.monotonic()-loop_start
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,config_sha256=sha(CFG),method=method),dest/'final.pt')
    result=dict(method=method,steps=step,frame_focus_exposures=int(order.size),training_questions=500,
          training_schedule_sha256=sha(out/'training-schedule.npy'),initial_relation_sha256=initial_relation_sha,
          trainable_parameters=parameter_count(model),projection_parameters=parameter_count(model.projection),relation_parameters=parameter_count(model.relation),
          train_loop_seconds=seconds,total_cache_train_save_seconds=time.monotonic()-start,
          peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
          final_loss=float(loss.detach()),curves=curves,all_loss_and_gradient_finite=True,final_step_only=True)
    save_json(dest/'training.json',result);print('TRAIN_COMPLETE',method,seconds,flush=True)


def evaluate(c,out):
    setup(c,True)
    if (out/'evaluation.json').exists():print('REUSE_EVALUATION',flush=True);return
    # Train-only rule selection saved before opening any development predictions.
    for method in ['P1','P2']:
        p=out/(method+'-calibration.json')
        if not p.exists():save_json(p,threshold_calibration(ROOT,c,method))
    calibrations={m:read_json(out/(m+'-calibration.json')) for m in ['P1','P2']}
    masks=torch.from_numpy(make_focus_masks(c)).unsqueeze(1).cuda()
    models={}
    for m in ['P0','P2plus','P3']:
        model=RelationProbe(1 if m=='P2plus' else 2048,c).cuda()
        ckpt=torch.load(out/'heads'/m/'final.pt',map_location='cuda',weights_only=True)
        assert ckpt['step']==1000 and ckpt['config_sha256']==sha(CFG)
        model.load_state_dict(ckpt['model'],strict=True);model.eval();models[m]=model
    rows=[];switch=[];inference=[]
    for s in c['sequences']:
        name=s['sequence'];data=np.load(out/'data'/f'{name}.npz');labels=data['labels'];boundary=data['boundary']
        dest=out/'predictions'/name;dest.mkdir(parents=True,exist_ok=True)
        predictions={m:np.zeros_like(labels) for m in ['P0','P1','P2','P2plus','P3']}
        depth=np.load(out/'features'/name/'depth.npy',mmap_mode='r')
        for method in ['P1','P2']:
            values=depth if method=='P2' else data['vda'];threshold=calibrations[method]['threshold']
            for t in range(25):
                for f,box in enumerate(focus_boxes(c)):
                    score=rule_score(values[t],box,method=='P2');p=np.ones(score.shape,np.uint8)
                    p[score>threshold]=0;p[score < -threshold]=2;predictions[method][t,f]=p
        for method in models:
            features=scene_features(c,out,name,method).cuda()
            torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():
                for t in range(25):
                    logits=models[method](features[t:t+1].expand(5,-1,-1,-1),masks)
                    assert torch.isfinite(logits).all()
                    predictions[method][t]=logits.argmax(1).cpu().numpy().astype(np.uint8)
            torch.cuda.synchronize();inference.append(dict(method=method,sequence=name,questions=125,seconds=time.monotonic()-start))
            del features
        for method,pred in predictions.items():
            np.save(dest/(method+'.npy'),pred)
            for region in ['all_valid','boundary','interior']:
                mask=None if region=='all_valid' else np.broadcast_to((boundary if region=='boundary' else ~boundary)[:,None],labels.shape)
                row=dict(method=method,sequence=name,source_family=s['source_family'],split=s['split'],region=region,
                         questions=int(sum((labels[t,f]<3).any() for t in range(25) for f in range(5))),**metrics(confusion(labels,pred,mask)))
                rows.append(row)
            count=0;changed=0;both_correct=0;pair_cm=np.zeros((9,9),np.int64)
            for a in range(5):
                for b in range(a+1,5):
                    ya,yb=labels[:,a],labels[:,b];pa,pb=pred[:,a],pred[:,b]
                    good=(ya<3)&(yb<3);m=good&(ya!=yb)
                    count+=int(m.sum());changed+=int((m&(pa!=pb)).sum());both_correct+=int((m&(pa==ya)&(pb==yb)).sum())
                    pair_cm+=np.bincount(9*(3*ya[good].astype(np.int64)+yb[good])+3*pa[good].astype(np.int64)+pb[good],minlength=81).reshape(9,9)
            switch.append(dict(method=method,sequence=name,split=s['split'],GT_changed_pixels_over10_focus_pairs=count,
                          prediction_changed_fraction=None if not count else changed/count,
                          both_endpoints_correct_fraction=None if not count else both_correct/count,
                          focus_transition_confusion_9x9=pair_cm.tolist(),note='same-frame focal-region switching, not temporal flicker or tracking'))
        print('EVALUATED',name,flush=True)
    summary=[]
    for method in predictions:
        for split in ['pilot_train','pilot_dev']:
            for region in ['all_valid','boundary','interior']:
                rs=[r for r in rows if r['method']==method and r['split']==split and r['region']==region]
                summary.append(dict(method=method,split=split,region=region,sources=len(rs),
                     Macro_F1=float(np.mean([r['Macro_F1'] for r in rs])),Balanced_Accuracy=float(np.mean([r['Balanced_Accuracy'] for r in rs])),
                     aggregation='equal mean of source metrics; each source confusion pools its frames/focuses'))
    get=lambda m,reg:next(r['Macro_F1'] for r in summary if r['method']==m and r['split']=='pilot_dev' and r['region']==reg)
    delta=get('P3','all_valid')-get('P2plus','all_valid');boundary_delta=get('P3','boundary')-get('P2plus','boundary')
    source_delta={s:next(r['Macro_F1'] for r in rows if r['method']=='P3' and r['sequence']==s and r['region']=='all_valid')-next(r['Macro_F1'] for r in rows if r['method']=='P2plus' and r['sequence']==s and r['region']=='all_valid') for s in ['cave_2','market_2']}
    decision=dict(P3_minus_P2plus_Macro_F1=delta,P3_minus_P2plus_boundary_Macro_F1=boundary_delta,
                  per_dev_source_delta=source_delta,source_directions_consistent=all(v>0 for v in source_delta.values()),
                  engineering_pass=bool(delta>=.02 and boundary_delta>=-.01),
                  criterion='source-equal dev Macro-F1 +0.02; boundary Macro-F1 no worse than -0.01; source direction reported separately',
                  independent_test=False,statistical_significance_claim=False,Stage1_started=False)
    result=dict(config_sha256=sha(CFG),methods=['P0','P1','P2','P2plus','P3'],completed_all5=True,
                per_source=rows,summary=summary,focus_switch=switch,calibrations=calibrations,decision=decision,head_inference=inference)
    save_json(out/'evaluation.json',result)
    save_json(ROOT/(c['record_prefix']+'-results.json'),result)
    for method in result['methods']:
        small=dict(method=method,config_sha256=sha(CFG),
                   per_source=[r for r in rows if r['method']==method],
                   summary=[r for r in summary if r['method']==method],
                   focus_switch=[r for r in switch if r['method']==method],
                   calibration=calibrations.get(method),
                   final_checkpoint='step1000' if method in models else 'frozen direct rule')
        save_json(ROOT/(c['record_prefix']+'-'+method+'.json'),small)
        write_csv(ROOT/(c['record_prefix']+'-'+method+'.csv'),small['per_source'])
    write_csv(ROOT/(c['record_prefix']+'-per-source.csv'),rows)
    write_csv(ROOT/(c['record_prefix']+'-summary.csv'),summary)
    write_csv(ROOT/(c['record_prefix']+'-focus-switch.csv'),switch)
    main=[]
    for method in predictions:
        r=dict(method=method)
        for s in ['cave_2','market_2']:
            r[s+'_Macro_F1']=next(v['Macro_F1'] for v in rows if v['method']==method and v['sequence']==s and v['region']=='all_valid')
        for key,reg in [('source_mean_Macro_F1','all_valid'),('boundary_Macro_F1','boundary')]:r[key]=get(method,reg)
        r['Balanced_Accuracy']=next(v['Balanced_Accuracy'] for v in summary if v['method']==method and v['split']=='pilot_dev' and v['region']=='all_valid')
        train_record=read_json(out/'heads'/method/'training.json') if method in models else None
        r['trainable_parameters']=train_record['trainable_parameters'] if train_record else 0
        r['projection_parameters']=train_record['projection_parameters'] if train_record else 0
        r['relation_parameters']=train_record['relation_parameters'] if train_record else 0
        main.append(r)
    write_csv(ROOT/(c['record_prefix']+'-main.csv'),main)
    save_json(out/'main.json',main);print('DECISION',json.dumps(decision),flush=True)


def report(c,out):
    if (ROOT/(c['record_prefix']+'-resources.json')).exists():
        assert (out/'report/index.html').exists()
        print('REUSE_REPORT',flush=True);return
    palette=np.array([[237,105,65],[100,194,108],[77,128,222],[160,160,160]],np.uint8)
    report=out/'report';report.mkdir(exist_ok=True)
    for s in c['sequences']:
        name=s['sequence'];data=np.load(out/'data'/f'{name}.npz');pred={m:np.load(out/'predictions'/name/(m+'.npy'),mmap_mode='r') for m in ['P0','P1','P2','P2plus','P3']}
        sheet=Image.new('RGB',(518*8,254*3),'white');draw=ImageDraw.Draw(sheet)
        for row,t in enumerate([0,12,24]):
            frame=s['frames'][t];rgb=Image.open(ROOT/c['raw_root']/'training/clean'/name/f'frame_{frame:04d}.png').convert('RGB').resize((518,224),Image.Resampling.BICUBIC)
            dr=ImageDraw.Draw(rgb);x,y,w,h=focus_boxes(c)[1];dr.rectangle([x,y,x+w-1,y+h-1],outline='yellow',width=2)
            y=data['labels'][t,1];valid=y<3;arrays=[rgb,Image.fromarray(palette[np.where(valid,y,3)]),Image.fromarray(np.uint8(valid)*255).convert('RGB')]
            arrays.extend(Image.fromarray(palette[np.where(valid,p[t,1],3)]) for p in pred.values())
            for col,(title,im) in enumerate(zip(['RGB focus1','GT','valid','P0 RGB','P1 VDA','P2 VGGT depth','P2+ learned depth','P3 VGGT layer17'],arrays)):
                sheet.paste(im,(518*col,row*254+30));draw.text((518*col+4,row*254+7),f'{name} f{frame} '+title,fill='black')
        sheet.save(report/(name+'.png'))
    html='<!doctype html><meta charset="utf-8"><title>Architecture V2 Stage0</title><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}</style><h1>Stage0：焦点相对三分类</h1><p>固定6来源、帧1/13/25、focus1；近=橙，相近=绿，远=蓝，无效=灰。不是虚化视频或视觉质量评分。</p>'
    main=read_json(out/'main.json');html+='<table><tr>'+''.join('<th>'+k+'</th>' for k in main[0])+'</tr>'
    for row in main:html+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in row.values())+'</tr>'
    html+='</table>'
    for s in c['sequences']:html+='<h2>'+s['sequence']+' / '+s['split']+'</h2><a href="'+s['sequence']+'.png"><img src="'+s['sequence']+'.png"></a>'
    (report/'index.html').write_text(html)
    vda_parameter_file=out/'VDA-parameters.json'
    if not vda_parameter_file.exists():
        # Reuse the already measured architecture count, not a new VDA forward.
        prior=read_json(ROOT/'03_实验记录/records/v2-stage0-20261029-resources.json')['VDA_parameters']
        save_json(vda_parameter_file,prior)
    resource=dict(extractions={k:read_json(out/(k+'-extraction.json')) for k in ['P0','VGGT']},
                  training={m:read_json(out/'heads'/m/'training.json') for m in ['P0','P2plus','P3']},
                  VDA_parameters=read_json(vda_parameter_file),
                  VDA_existing_cost=read_json(ROOT/c['vda_root']/'summary.json'),
                  cost_limit='actual current feature/heading costs; shared VGGT extraction not counted 3 times, no end-to-end bokeh cost comparison')
    save_json(ROOT/(c['record_prefix']+'-resources.json'),resource)


def main():
    global CFG
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','extract-vggt','extract-p0','train-p0','train-p2plus','train-p3','evaluate','report','all'])
    parser.add_argument('--config',default=str(CFG),help='For independent reproduction copy frozen config and change output/record_prefix only.')
    args=parser.parse_args();CFG=Path(args.config).resolve()
    c=read_json(CFG);out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True);setup(c)
    stages=['prepare','extract-vggt','extract-p0','train-p0','train-p2plus','train-p3','evaluate','report']
    if args.stage=='all':
        launch=dict(pid=os.getpid(),UTC=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config_sha256=sha(CFG),
                  nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True))
        if (out/'launch.json').exists():
            assert read_json(out/'launch.json')['config_sha256']==sha(CFG)
            print('RESUME_EXISTING_LAUNCH',flush=True)
        else:save_json(out/'launch.json',launch)
        for stage in stages:
            subprocess.run([sys.executable,__file__,stage,'--config',str(CFG)],check=True,cwd=ROOT)
        return
    if args.stage=='prepare':prepare(c,out)
    elif args.stage.startswith('extract-'):extract(c,out,'VGGT' if args.stage=='extract-vggt' else 'P0')
    elif args.stage.startswith('train-'):train(c,out,{'train-p0':'P0','train-p2plus':'P2plus','train-p3':'P3'}[args.stage])
    elif args.stage=='evaluate':evaluate(c,out)
    elif args.stage=='report':report(c,out)


if __name__=='__main__':main()
