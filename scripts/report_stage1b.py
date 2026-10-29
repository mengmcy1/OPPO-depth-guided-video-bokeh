"""Stage1b verification and fixed-scene report. Never selects checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.focus_relative_probe import read_json,save_json,sha
from scripts import train_stage1a_generation as old
from scripts.prepare_stage1b import C,OUT,PREFIX,blur_bank,observable_target,signed_base_coc
CFG=ROOT/'configs/v2-stage1b-20261029.json'


def prediction(phase,scene,arm):
    if phase=='ray' and arm in C['ray_existing_arms']:
        return np.load(old.OUT/'predictions'/scene/f"{C['ray_existing_arms'][arm]}-f0-k32.npy")
    return np.load(OUT/phase/'predictions'/scene/f'{arm}-f0-k32.npy')


def report():
    import imageio_ffmpeg
    dest=OUT/'report';dest.mkdir(exist_ok=True)
    html='<meta charset="utf-8"><title>Stage1b</title><h1>Stage1b 对照</h1><p>原开发/原测试都已见，非新独立验证。固定焦点0、K32，完整25帧；Oracle有GT几何权限。几何分类/生成数值不替用户判断观感。</p>'
    for phase in ['observable','ray']:
        rec=read_json(OUT/phase/'evaluation.json')
        html+=f'<h2>{phase}</h2><p>'+('可观测Gaussian目标：所有目标像素来自输入RGB，不模拟遮挡光学。' if phase=='observable' else '原分层光线积分目标。DINO/VGGT为相同适配器，不同冻结骨干特征。')+'</p>'
        html+='<table border="1"><tr><th>来源组</th><th>方法</th><th>PSNR</th><th>MAE</th><th>SSIM</th></tr>'
        for r in rec['summary']:
            if r['region']=='all' and r['split']!='train' and not r['kind'].endswith('_zero'):
                html+=f"<tr><td>{r['split']}</td><td>{r['kind']}</td><td>{r['PSNR']:.3f}</td><td>{r['MAE']:.5f}</td><td>{r['SSIM']:.4f}</td></tr>"
        html+='</table>'
        if phase=='observable':arms=['RGB','OracleLogSmall','OracleCoCSmall','OracleLogStrong','OracleCoCStrong']
        else:arms=['DepthStrong' if 'DepthStrong' in rec['arms'] else 'DepthSmall','DINO','VGGT','OracleCoCStrong' if 'OracleCoCStrong' in rec['arms'] else 'OracleLogSmall']
        for s in [s for s in old.C['scenes'] if s['split']!='train']:
            name=s['id'];base=old.OUT/'data'/name
            target=OUT/'data'/name/'observable.npy' if phase=='observable' else base/'targets.npy'
            arrays=[np.load(base/'rgb.npy').astype(np.float32)/255,np.load(target)[:,0,1]]+[prediction(phase,name,a) for a in arms]
            labels=['Input RGB / focus0','Reference']+arms
            height=((len(labels)+2)//3)*282;frames=[];sheet=Image.new('RGB',(1344,height*3),'white')
            for t in range(25):
                canvas=Image.new('RGB',(1344,height),'white');draw=ImageDraw.Draw(canvas)
                for j,(a,label) in enumerate(zip(arrays,labels)):
                    xx=j%3*448;yy=j//3*282;im=Image.fromarray(np.uint8(np.round(np.clip(a[t].transpose(1,2,0),0,1)*255)))
                    if j==0:
                        x,y,w,h=old.boxes()[0];ImageDraw.Draw(im).rectangle([x,y,x+w-1,y+h-1],outline='yellow',width=1)
                    canvas.paste(im,(xx,yy+30));draw.text((xx+5,yy+8),f'{name} f{t} {label}',fill='black')
                frames.append(np.asarray(canvas))
                if t in [0,12,24]:sheet.paste(canvas,(0,[0,12,24].index(t)*height))
            stem=f'{phase}-{name}';sheet.save(dest/f'{stem}.png')
            writer=imageio_ffmpeg.write_frames(str(dest/f'{stem}.mp4'),(1344,height),fps=20,codec='libx264',pix_fmt_in='rgb24',output_params=['-crf','16','-movflags','+faststart'],macro_block_size=1)
            writer.send(None)
            for frame in frames:writer.send(frame)
            writer.close()
            html+=f'<article><h3>{phase} {name} / {s["split"]}</h3><video controls style="max-width:100%" src="{stem}.mp4"></video><p>25帧，五帧模型分组；到结尾停止，重新播放不是连续过渡。</p>'
            for speed in [1,.5,.25]:html+=f'<button onclick="this.closest(\'article\').querySelector(\'video\').playbackRate={speed}">{speed}×</button>'
            html+=f'<p><a href="{stem}.png">0/12/24帧原尺寸对照</a></p></article>'
    (dest/'index.html').write_text(html);print('REPORT_DONE',dest,flush=True)


def verify():
    old.setup();base_train=read_json(old.OUT/'heads/RGB/training.json');training=[];endpoints=0;replays=0;old_reuse=0
    targets=read_json(OUT/'targets.json');assert targets['config_sha256']==sha(CFG)
    # Reconstruct one target from the actual normalized half-precision CoC
    # condition, proving hidden layers are not needed by this diagnostic.
    scene=old.C['scenes'][0]['id'];p=old.OUT/'data'/scene
    rgb=np.load(p/'rgb.npy')[0].transpose(1,2,0).astype(np.float32)/255;z=np.load(p/'GT_depth.npy')
    base=signed_base_coc(z)[0,0];encoded=(base/C['observable_target']['control_normalizer_pixels']).astype(np.float16)
    recon=observable_target(blur_bank(rgb),encoded.astype(np.float32)*32*C['observable_target']['control_normalizer_pixels'])
    target=np.load(OUT/'data'/scene/'observable.npy')[0,0,1].transpose(1,2,0).astype(np.float32)
    err=float(np.max(abs(recon-target)));assert err<.0003,err
    for s in old.C['scenes']:
        a=np.load(OUT/'data'/s['id']/'observable.npy',mmap_mode='r');assert a.shape==(25,2,2,3,252,448) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
        ft=np.load(OUT/'data'/s['id']/'DINO.npy',mmap_mode='r');assert ft.shape==(25,2048,21,37) and np.isfinite(ft).all()
    for phase in ['observable','ray']:
        rec=read_json(OUT/phase/'evaluation.json');assert rec['config_sha256']==sha(CFG)
        replays+=sum(bool(r.get('final_group_reloaded_bitwise')) for r in rec['checks'])
        old_reuse+=sum(bool(r.get('reused_frozen_stage1a_predictions')) for r in rec['checks'])
        for arm in rec['arms']:
            if phase=='ray' and arm in C['ray_existing_arms']:
                old_arm=C['ray_existing_arms'][arm];tr=read_json(old.OUT/'heads'/old_arm/'training.json')
                assert sha(old.OUT/'heads'/old_arm/'final.pt')==tr['checkpoint_sha256'];continue
            tr=read_json(OUT/phase/'heads'/arm/'training.json');training.append(tr)
            assert tr['steps']==2000 and tr['schedule_sha256']==base_train['schedule_sha256'] and tr['initial_trunk_sha256']==base_train['initial_trunk_sha256']
            ck=torch.load(OUT/phase/'heads'/arm/'final.pt',map_location='cpu',weights_only=True)
            assert ck['config_sha256']==sha(CFG) and ck['step']==2000
            assert all(int(v['step'])==2000 for v in ck['optimizer']['state'].values())
            assert all(torch.isfinite(v).all() for v in ck['model'].values())
        rows=list(csv.DictReader(open(str(PREFIX)+f'-{phase}-frames.csv')))
        lookup={(r['scene'],r['kind'],int(r['focus']),float(r['K']),int(r['frame']),r['region']):r for r in rows}
        for s in [s for s in old.C['scenes'] if s['split']!='train']:
            name=s['id'];target_path=OUT/'data'/name/'observable.npy' if phase=='observable' else old.OUT/'data'/name/'targets.npy'
            y=np.load(target_path,mmap_mode='r')
            for arm in rec['arms']:
                for f in range(2):
                    for ki,K in enumerate(old.C['data']['K']):
                        if phase=='ray' and arm in C['ray_existing_arms']:path=old.OUT/'predictions'/name/f"{C['ray_existing_arms'][arm]}-f{f}-k{int(K)}.npy"
                        else:path=OUT/phase/'predictions'/name/f'{arm}-f{f}-k{int(K)}.npy'
                        pred=np.load(path,mmap_mode='r');assert pred.shape==(25,3,252,448) and np.isfinite(pred).all()
                        for t in [0,24]:
                            e=pred[t,:,12:-12,12:-12].astype(np.float32)-y[t,f,ki,:,12:-12,12:-12].astype(np.float32)
                            psnr=-10*np.log10(max(float((e*e).mean()),1e-12));mae=float(abs(e).mean());r=lookup[(name,arm,f,K,t,'all')]
                            assert abs(psnr-float(r['PSNR']))<1e-4 and abs(mae-float(r['MAE']))<1e-7
                            endpoints+=1
        # Freeze focal/strength summaries as auxiliary outputs, no selection.
        strata=[]
        for split in ['dev','test']:
            for arm in ['Identity']+rec['arms']:
                for f in range(2):
                    for K in old.C['data']['K']:
                        per=[]
                        for s in old.C['scenes']:
                            if s['split']!=split:continue
                            rr=[r for r in rows if (r['scene'],r['kind'],r['region'],int(r['focus']),float(r['K']))==(s['id'],arm,'all',f,K)]
                            per.append({m:float(np.mean([float(r[m]) for r in rr])) for m in ['PSNR','MAE','SSIM']})
                        strata.append(dict(split=split,kind=arm,focus=f,K=K,**{m:float(np.mean([r[m] for r in per])) for m in ['PSNR','MAE','SSIM']}))
        old.write_csv(str(PREFIX)+f'-{phase}-focus-strength.csv',strata)
    gate=read_json(OUT/'observable/evaluation.json')['gate']
    assert read_json(OUT/'ray/evaluation.json')['arms']==list(C['ray_existing_arms'])+(C['ray_new_arms_if_gate'] if gate['passed'] else ['DINO'])
    a={r['arm']:r for r in training if r['phase']=='observable'}
    assert a['OracleLogSmall']['initial_adapter_sha256']==a['OracleCoCSmall']['initial_adapter_sha256']
    assert a['OracleLogStrong']['initial_adapter_sha256']==a['OracleCoCStrong']['initial_adapter_sha256']
    record=dict(status='PASS',config_sha256=sha(CFG),prediction_metric_endpoint_checks=endpoints,checkpoint_final_group_replays=replays,
        reused_stage1a_source_arm_checks=old_reuse,old_checkpoints_unchanged=True,matching_initial_trunk_and_schedule=True,
        paired_log_CoC_adapter_initialization_equal=True,observable_target_reconstructed_from_actual_condition_max_error=err,
        conditional_branch_respected=True,observable_gate=gate,training=training,targets=targets,DINO=read_json(OUT/'DINO.json'),
        initial_checks=read_json(OUT/'implementation-check.json'),no_new_independent_test=True,visual_review='user pending')
    save_json(str(PREFIX)+'-verification.json',record);print('VERIFIED',endpoints,'metrics;',replays,'replays; models',len(training),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['report','verify']);a=p.parse_args();{'report':report,'verify':verify}[a.phase]()
