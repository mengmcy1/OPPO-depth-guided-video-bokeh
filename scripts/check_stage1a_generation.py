"""Verify frozen generation pilot products without changing or selecting models."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.focus_relative_probe import read_json,save_json,sha
from scripts.prepare_stage1a_generation import C,CFG,OUT,PREFIX,boxes
from scripts.check_benchmark_source_isolation import collect_development


def main():
    old={r['source_id'] for r in collect_development()[0]};ids=[];focus_stats=[]
    for s in C['scenes']:
        for key in ['foreground_source_id','background_source_id']:
            assert s[key] not in old and s[key] not in ids;ids.append(s[key])
        d=OUT/'data'/s['id'];meta=read_json(d/'prepared.json')
        rgb=np.load(d/'rgb.npy',mmap_mode='r');target=np.load(d/'targets.npy',mmap_mode='r');z=np.load(d/'GT_depth.npy')
        assert rgb.shape==(25,3,252,448) and target.shape==(25,2,2,3,252,448)
        assert target.dtype==np.float16 and np.isfinite(target).all() and target.min()>=0 and target.max()<=1
        for t in [0,24]:assert np.array_equal(rgb[t],np.array(Image.open(d/'rgb'/f'{t:05d}.png')).transpose(2,0,1))
        recomputed=np.array([[np.median(a[y:y+h,x:x+w]) for x,y,w,h in boxes()] for a in z])
        assert np.array_equal(recomputed,np.array(meta['focus_depth_m']))
        focus_stats.append(dict(scene=s['id'],split=s['split'],frames_with_distinct_focus_depths=int((recomputed[:,0]!=recomputed[:,1]).sum()),
            total_frames=25,focus_depth_values=np.unique(recomputed,axis=0).tolist()))
    results=read_json(OUT/'evaluation.json');training=results['training']
    assert len({r['initial_trunk_sha256'] for r in training})==1 and len({r['schedule_sha256'] for r in training})==1
    assert all(r['config_sha256']==sha(CFG) and r['steps']==2000 for r in training)
    for kind in C['groups']:
        ck=torch.load(OUT/'heads'/kind/'final.pt',map_location='cpu',weights_only=True)
        assert ck['step']==2000 and ck['config_sha256']==sha(CFG)
        assert all(torch.isfinite(v).all() for v in ck['model'].values())
        assert all(int(v['step'])==2000 for v in ck['optimizer']['state'].values())
    rows=list(csv.DictReader(open(str(PREFIX)+'-frames.csv')))
    bykey={(r['scene'],r['kind'],int(r['focus']),float(r['K']),int(r['frame']),r['region']):r for r in rows}
    checked=0;errors=[]
    for s in C['scenes']:
        if s['split']=='train':continue
        target=np.load(OUT/'data'/s['id']/'targets.npy',mmap_mode='r')
        for kind in C['groups']:
            for fi in range(2):
                for ki,K in enumerate(C['data']['K']):
                    pred=np.load(OUT/'predictions'/s['id']/f'{kind}-f{fi}-k{int(K)}.npy',mmap_mode='r')
                    assert pred.shape==(25,3,252,448) and np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
                    for t in [0,24]:
                        e=pred[t,:,12:-12,12:-12].astype(np.float32)-target[t,fi,ki,:,12:-12,12:-12].astype(np.float32)
                        psnr=float(-10*np.log10(max(float((e*e).mean()),1e-12)));mae=float(np.abs(e).mean())
                        r=bykey[(s['id'],kind,fi,K,t,'all')]
                        assert abs(psnr-float(r['PSNR']))<1e-4 and abs(mae-float(r['MAE']))<1e-7
                        checked+=1;errors.append(abs(psnr-float(r['PSNR'])))
    # Explicitly distinguish numeric token margin from a valid full comparison.
    conclusion=dict(status='COMPLETE_ORACLE_SANITY_FAILED',
        numeric_tokens_advantage=results['decisions'],
        overall_comparison_validated=False,
        reason='Oracle fails predeclared +1dB above identity; final-depth arms not yet shown to exploit even correct geometry',
        qualified_finding='Tokens help this small generator under this frozen budget, but geometry-vs-appearance and adapter capacity/optimization are confounded',
        training_fit=read_json(str(PREFIX)+'-training-fit-diagnostic.json')['summary'],
        next='No automatic extra training. Next design needs a usable oracle/depth interface and a matched frozen visual-feature condition control.',
        no_full_ATB_or_real_video_claim=True)
    save_json(str(PREFIX)+'-conclusion.json',conclusion)
    verification=dict(status='PASS',config_sha256=sha(CFG),source_ids=ids,source_overlap=0,old_registered_overlap=0,
        all_target_arrays_finite=True,known_geometry_focus_matches_renderer=True,PNG_vs_model_input_checked=True,
        matching_trunk_initialization=True,matching_schedule=True,all_optimizer_steps=2000,
        prediction_metric_endpoint_checks=checked,max_PSNR_recompute_difference=max(errors),
        checkpoint_reload_and_zeroK_checks=results['checks'],focus_coverage=focus_stats,
        renderer_checks=read_json(OUT/'renderer-check.json'),renderer_costs=read_json(OUT/'rendered.json'),
        feature_costs=read_json(OUT/'features.json'),training_costs=training,
        recovery_notes=['NumPy scalar JSON issue fixed before training','gen09 starts at native frame26 by earliest25 nonempty-alpha rule, IDs unchanged',
            'Training env lacks skimage/scipy; original evaluation stopped before learned predictions. OpenCV uniform7 SSIM verified against installed skimage0.24 on3 fixed training frames, max difference8.51e-8; weights/data unchanged'],
        videos='four held-out source clips,25 frames each; user visual review pending')
    save_json(str(PREFIX)+'-verification.json',verification)
    # Required focal/strength stratification, no model or threshold selection.
    strata=[]
    for split in ['dev','test']:
        for kind in ['Identity']+C['groups']:
            for fi in range(2):
                for K in C['data']['K']:
                    src=[]
                    for s in C['scenes']:
                        if s['split']!=split:continue
                        rr=[r for r in rows if r['scene']==s['id'] and r['kind']==kind and r['region']=='all' and int(r['focus'])==fi and float(r['K'])==K]
                        src.append({m:float(np.mean([float(r[m]) for r in rr])) for m in ['PSNR','MAE','SSIM']})
                    strata.append(dict(split=split,kind=kind,focus=fi,K=K,**{m:float(np.mean([r[m] for r in src])) for m in ['PSNR','MAE','SSIM']}))
    from scripts.train_stage1a_generation import write_csv
    write_csv(str(PREFIX)+'-focus-strength.csv',strata)
    print('PASS',checked,'endpoint metrics,16 checkpoint group replays,source/target/schedule checks')


if __name__=='__main__':main()
