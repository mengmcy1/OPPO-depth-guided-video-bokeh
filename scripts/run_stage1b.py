"""Frozen Stage1b factorial diagnosis and DINO control; no Stage1a mutations."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.focus_relative_probe import read_json,save_json,sha
from src.stage1b_probe import make_model
from scripts import train_stage1a_generation as old
from scripts.prepare_stage1b import C,OUT,PREFIX,signed_base_coc
CFG=ROOT/'configs/v2-stage1b-20261029.json'


def cache(scenes,phase,arm):
    xs=[];ys=[];cs=[]
    for s in scenes:
        p=old.OUT/'data'/s['id'];xs.append(torch.from_numpy(np.load(p/'rgb.npy')))
        target=OUT/'data'/s['id']/'observable.npy' if phase=='observable' else p/'targets.npy'
        ys.append(torch.from_numpy(np.load(target)))
        if arm in ['DINO','VGGT']:
            path=OUT/'data'/s['id']/'DINO.npy' if arm=='DINO' else p/'features/tokens.npy'
            cs.append(torch.from_numpy(np.load(path)))
        elif arm!='RGB':
            depth=np.load(p/('features/depth.npy' if arm.startswith('Depth') else 'GT_depth.npy'))
            if 'CoC' in arm:
                c=signed_base_coc(depth);val=np.stack([c,abs(c)],2)/C['observable_target']['control_normalizer_pixels']
            else:val=old.depth_conditions(depth)
            cs.append(torch.from_numpy(val.astype(np.float16)))
    return dict(rgb=torch.cat(xs).cuda(),target=torch.cat(ys).cuda(),condition=torch.cat(cs).cuda() if cs else None)


def batch(data,q,arm,masks):
    kind='RGB' if arm=='RGB' else 'Tokens' if arm in ['DINO','VGGT'] else 'Depth'
    x,m,k,c,y=old.batch(data,q,kind,masks)
    if 'CoC' in arm:c=c*k[:,None,None,None,None]
    return x,m,k,c,y


def train(phase,arm):
    old.setup(True);torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed'])
    dest=OUT/phase/'heads'/arm;dest.mkdir(exist_ok=True,parents=True)
    if (dest/'training.json').exists():print('REUSE_TRAIN',phase,arm);return
    assert phase=='observable' and arm in C['observable_arms'] or phase=='ray' and arm in C['ray_new_arms_if_gate']
    start=time.monotonic();data=cache([s for s in old.C['scenes'] if s['split']=='train'],phase,arm)
    masks=old.focus_masks();model=make_model(arm,C['seed']).cuda()
    trunk=old.state_digest({n:v for n,v in model.state_dict().items() if not n.startswith(('proj.','interaction.'))})
    original=read_json(old.OUT/'heads/RGB/training.json');assert trunk==original['initial_trunk_sha256']
    initial_adapter=old.state_digest({n:v for n,v in model.state_dict().items() if n.startswith(('proj.','interaction.'))})
    opt=torch.optim.AdamW(model.parameters(),lr=C['train']['lr'],weight_decay=C['train']['weight_decay'])
    order=np.load(old.OUT/'schedule.npy');questions=np.load(old.OUT/'questions.npy');assert order.shape==(4000,)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.monotonic();curves=[]
    for step,ix in enumerate(order.reshape(2000,2),1):
        x,m,k,c,y=batch(data,questions[ix],arm,masks);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x,m,k,c)
        loss=F.l1_loss(p.float(),y.float());assert torch.isfinite(loss)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);opt.step()
        if step==1 or step%100==0:
            curves.append(dict(step=step,L1=float(loss.detach()),gradient_norm=float(grad)))
            print('TRAIN',phase,arm,step,round(float(loss.detach()),6),round(time.monotonic()-t0,1),flush=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-t0
    assert not (dest/'final.pt').exists()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=2000,phase=phase,arm=arm,config_sha256=sha(CFG)),dest/'final.pt')
    rec=dict(phase=phase,arm=arm,steps=2000,clip_exposures=4000,frame_exposures=20000,
        config_sha256=sha(CFG),initial_trunk_sha256=trunk,initial_adapter_sha256=initial_adapter,schedule_sha256=sha(old.OUT/'schedule.npy'),
        parameters=sum(p.numel() for p in model.parameters()),adapter_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith(('proj.','interaction.'))),
        training_seconds=elapsed,load_train_save_seconds=time.monotonic()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        curves=curves,checkpoint_sha256=sha(dest/'final.pt'),torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    save_json(dest/'training.json',rec);save_json(str(PREFIX)+f'-{phase}-{arm}-train.json',rec)
    print('TRAIN_COMPLETE',phase,arm,elapsed,flush=True)


def model_load(phase,arm):
    if phase=='ray' and arm in C['ray_existing_arms']:
        return old.model_load(C['ray_existing_arms'][arm])
    ck=torch.load(OUT/phase/'heads'/arm/'final.pt',map_location='cuda',weights_only=True)
    assert ck['step']==2000 and ck['config_sha256']==sha(CFG)
    model=make_model(arm,C['seed']).cuda().eval();model.load_state_dict(ck['model'],strict=True);return model


def summarize(rows):
    source=[];summary=[]
    for scene in sorted({r['scene'] for r in rows}):
        for arm in sorted({r['kind'] for r in rows}):
            for region in ['all','boundary','foreground_core','background_core']:
                rr=[r for r in rows if (r['scene'],r['kind'],r['region'])==(scene,arm,region)]
                if not rr:continue
                rec=dict(scene=scene,split=rr[0]['split'],kind=arm,region=region)
                for metric in ['PSNR','MAE','SSIM']:
                    vals=[r[metric] for r in rr if r.get(metric) is not None];rec[metric]=float(np.mean(vals)) if vals else None
                source.append(rec)
    for split in sorted({r['split'] for r in source}):
        for arm in sorted({r['kind'] for r in source}):
            for region in ['all','boundary','foreground_core','background_core']:
                rr=[r for r in source if (r['split'],r['kind'],r['region'])==(split,arm,region)]
                if not rr:continue
                rec=dict(split=split,kind=arm,region=region,sources=len(rr))
                for metric in ['PSNR','MAE','SSIM']:
                    vals=[r[metric] for r in rr if r.get(metric) is not None];rec[metric]=float(np.mean(vals)) if vals else None
                summary.append(rec)
    return source,summary


def evaluate(phase):
    old.setup(True);dest=OUT/phase;dest.mkdir(exist_ok=True)
    if (dest/'evaluation.json').exists():print('REUSE_EVALUATION',phase);return
    if phase=='observable':arms=C['observable_arms'];scenes=old.C['scenes']
    else:
        gate=read_json(OUT/'observable/evaluation.json')['gate']['passed']
        arms=list(C['ray_existing_arms'])+(C['ray_new_arms_if_gate'] if gate else ['DINO'])
        scenes=[s for s in old.C['scenes'] if s['split']!='train']
    masks=old.focus_masks();rows=[];checks=[];cost=[]
    for s in scenes:
        name=s['id'];p=old.OUT/'data'/name
        target_path=OUT/'data'/name/'observable.npy' if phase=='observable' else p/'targets.npy'
        targets=np.load(target_path);rgb=np.load(p/'rgb.npy').astype(np.float32)/255
        regions=old.region_masks(np.load(p/'alpha.npy').astype(np.float32));pred_dir=dest/'predictions'/name
        if s['split']!='train':pred_dir.mkdir(parents=True,exist_ok=True)
        for f in range(2):
            for ki,K in enumerate(old.C['data']['K']):
                for r in old.score_video(rgb,targets[:,f,ki],regions,s['split']!='train'):
                    rows.append(dict(scene=name,split=s['split'],kind='Identity',focus=f,K=K,**r))
        for arm in arms:
            if phase=='ray' and arm in C['ray_existing_arms']:
                old_arm=C['ray_existing_arms'][arm]
                for f in range(2):
                    for ki,K in enumerate(old.C['data']['K']):
                        path=old.OUT/'predictions'/name/f'{old_arm}-f{f}-k{int(K)}.npy';pred=np.load(path)
                        for r in old.score_video(pred,targets[:,f,ki],regions):rows.append(dict(scene=name,split=s['split'],kind=arm,focus=f,K=K,**r))
                checks.append(dict(scene=name,arm=arm,reused_frozen_stage1a_predictions=True));continue
            model=model_load(phase,arm);data=cache([s],phase,arm);start=time.monotonic()
            for f in range(2):
                for ki,K in enumerate(old.C['data']['K']):
                    preds=[];zero=[]
                    with torch.inference_mode():
                        for g in range(5):
                            x,m,k,c,y=batch(data,np.array([[0,g,f,ki]]),arm,masks)
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                pp=model(x,m,k,c);identity=model(x,m,torch.zeros_like(k),c)
                                assert torch.equal(identity,x)
                                if s['split']!='train' and arm!='RGB':zero.append(model(x,m,k,c,zero_condition=True)[0].cpu().numpy())
                            assert torch.isfinite(pp).all();preds.append(pp[0].cpu().numpy())
                    pred=np.concatenate(preds).astype(np.float16)
                    if s['split']!='train':np.save(pred_dir/f'{arm}-f{f}-k{int(K)}.npy',pred)
                    for r in old.score_video(pred,targets[:,f,ki],regions,s['split']!='train'):
                        rows.append(dict(scene=name,split=s['split'],kind=arm,focus=f,K=K,**r))
                    if zero:
                        za=np.concatenate(zero).astype(np.float16);np.save(pred_dir/f'{arm}-zero-f{f}-k{int(K)}.npy',za)
                        for r in old.score_video(za,targets[:,f,ki],regions,False):rows.append(dict(scene=name,split=s['split'],kind=arm+'_zero',focus=f,K=K,**r))
            clone=model_load(phase,arm)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):replay=clone(x,m,k,c)[0].cpu().numpy().astype(np.float16)
            assert np.array_equal(replay,pred[-5:])
            checks.append(dict(scene=name,arm=arm,zeroK_identity=True,final_group_reloaded_bitwise=True))
            cost.append(dict(scene=name,arm=arm,evaluation_seconds=time.monotonic()-start));del clone,model,data
            print('EVALUATED',phase,name,arm,flush=True)
    source,summary=summarize(rows)
    def metric(split,arm,region='all',key='PSNR'):
        return next(r[key] for r in summary if (r['split'],r['kind'],r['region'])==(split,arm,region))
    if phase=='observable':
        deltas={arm:{split:metric(split,arm)-metric(split,'Identity') for split in ['train','dev','test']} for arm in ['OracleCoCSmall','OracleCoCStrong']}
        passed=[arm for arm,value in deltas.items() if value['train']>=1 and value['dev']>=1]
        gate=dict(passed=bool(passed),qualifying_arms=passed,CoC_minus_Identity_dB=deltas,test_not_used_for_gate=True)
    else:
        gate={split:dict(VGGT_minus_DINO_dB=metric(split,'VGGT')-metric(split,'DINO'),
            VGGT_boundary_MAE_relative_increase=metric(split,'VGGT','boundary','MAE')/metric(split,'DINO','boundary','MAE')-1) for split in ['dev','test']}
    result=dict(phase=phase,arms=arms,source=source,summary=summary,gate=gate,checks=checks,cost=cost,config_sha256=sha(CFG),
        legacy_eval_sources_already_seen=True,no_model_selection=True,not_full_ATB_system=True)
    save_json(dest/'evaluation.json',result);save_json(str(PREFIX)+f'-{phase}-results.json',result)
    old.write_csv(str(PREFIX)+f'-{phase}-frames.csv',rows);old.write_csv(str(PREFIX)+f'-{phase}-per-source.csv',source);old.write_csv(str(PREFIX)+f'-{phase}-summary.csv',summary)
    print('GATE',phase,gate,flush=True)


def run_ray():
    gate=read_json(OUT/'observable/evaluation.json')['gate']['passed']
    selected=C['ray_new_arms_if_gate'] if gate else ['DINO']
    print('RAY_SELECTED',selected,'observable_gate',gate,flush=True)
    for arm in selected:
        with (ROOT/'logs/v2-stage1b-20261029'/f'ray-{arm}.log').open('ab') as log:
            subprocess.run([sys.executable,__file__,'train','--phase','ray','--arm',arm],check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    evaluate('ray')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate','run-ray']);p.add_argument('--phase',choices=['observable','ray']);p.add_argument('--arm');a=p.parse_args()
    if a.action=='train':train(a.phase,a.arm)
    elif a.action=='evaluate':evaluate(a.phase)
    else:run_ray()
