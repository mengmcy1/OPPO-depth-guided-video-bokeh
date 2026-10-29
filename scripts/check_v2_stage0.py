"""Verify saved Stage0 counts, train-only calibration, checkpoint and predictions."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.focus_relative_probe import (read_json,save_json,sha,confusion,metrics,
                                    RelationProbe,make_focus_masks)
from scripts.run_v2_stage0 import scene_features


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/v2-stage0-20261029.json'));args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=Path(args.config).resolve();c=read_json(cfg);out=ROOT/c['output'];ev=read_json(out/'evaluation.json')
    manifest=read_json(ROOT/(c['record_prefix']+'-manifest.json'))
    assert ev['config_sha256']==manifest['config_sha256']==sha(cfg)
    train={r['source_family'] for r in manifest['questions'] if r['split']=='pilot_train'}
    dev={r['source_family'] for r in manifest['questions'] if r['split']=='pilot_dev'}
    assert not train&dev and len(train)==4 and len(dev)==2
    assert len(manifest['questions'])==750 and all(r['valid_problem'] for r in manifest['questions'])
    assert len(ev['per_source'])==90
    checks=[]
    for s in c['sequences']:
        name=s['sequence'];data=np.load(out/'data'/f'{name}.npz');y=data['labels'];boundary=data['boundary']
        assert y.shape==(25,5,224,518) and np.isin(y,[0,1,2,255]).all()
        for method in ev['methods']:
            p=np.load(out/'predictions'/name/(method+'.npy'),mmap_mode='r');assert p.shape==y.shape and p.dtype==np.uint8 and (p<3).all()
            for region in ['all_valid','boundary','interior']:
                support=None if region=='all_valid' else np.broadcast_to((boundary if region=='boundary' else ~boundary)[:,None],y.shape)
                cm=confusion(y,p,support)
                row=next(r for r in ev['per_source'] if r['method']==method and r['sequence']==name and r['region']==region)
                assert cm.tolist()==row['confusion']
                assert metrics(cm)['Macro_F1']==row['Macro_F1']
            if method in ['P1','P2'] and s['split']=='pilot_train':
                row=next(r for r in ev['per_source'] if r['method']==method and r['sequence']==name and r['region']=='all_valid')
                assert row['confusion']==ev['calibrations'][method]['per_source'][name]['confusion']
            checks.append(dict(sequence=name,method=method,questions=125,CM_recomputed_from_saved_classes=True))
        with Image.open(out/'report'/f'{name}.png') as im:assert im.size==(4144,762)
    training={m:read_json(out/'heads'/m/'training.json') for m in ['P0','P2plus','P3']}
    assert len({r['initial_relation_sha256'] for r in training.values()})==1
    assert len({r['training_schedule_sha256'] for r in training.values()})==1
    assert all(r['steps']==1000 and r['frame_focus_exposures']==8000 for r in training.values())
    masks=torch.from_numpy(make_focus_masks(c)).unsqueeze(1).cuda();replays=[]
    for method in training:
        model=RelationProbe(1 if method=='P2plus' else 2048,c).cuda()
        ckpt=torch.load(out/'heads'/method/'final.pt',map_location='cuda',weights_only=True)
        assert ckpt['step']==1000 and ckpt['config_sha256']==ev['config_sha256']
        assert all(int(v['step'])==1000 for v in ckpt['optimizer']['state'].values())
        model.load_state_dict(ckpt['model'],strict=True);model.eval()
        for name in ['cave_2','market_2']:
            feature=scene_features(c,out,name,method)[0:1].cuda()
            with torch.inference_mode():p=model(feature.expand(5,-1,-1,-1),masks).argmax(1).cpu().numpy()[1]
            saved=np.load(out/'predictions'/name/(method+'.npy'),mmap_mode='r')[0,1]
            assert np.array_equal(p,saved)
            replays.append(dict(method=method,sequence=name,frame=1,focus_id=1,saved_classes_equal=True))
            del feature
        del model,ckpt
    result=dict(config_sha256=ev['config_sha256'],completed_methods=ev['methods'],source_split_disjoint=True,
                all750_questions_valid=True,all30_source_method_predictions=checks,all90_confusions_verified=True,
                rule_calibrations_train_only_verified=True,common_relation_initialization=True,common_training_schedule=True,
                all3_final_checkpoints_step1000=True,checkpoint_replays=replays,all6_comparison_figures_dimensions_correct=True,
                implementation_notes=read_json(out/'implementation-checks.json'),
                scope='code/numeric checks only; no perceptual bokeh judgment or claim of temporal consistency')
    save_json(ROOT/(c['record_prefix']+'-verification.json'),result)
    save_json(out/'verification.json',result)
    print('STAGE0_ALL_VERIFICATION_PASS',flush=True)


if __name__=='__main__':main()
