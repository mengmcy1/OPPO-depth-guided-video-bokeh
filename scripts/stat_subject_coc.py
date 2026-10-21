"""CPU input-only subject defocus statistics and preregistered selection.

The actual public loader is observed before inplace Normalize. No neural
models are loaded, and held-out videos or model predictions are never opened.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
import torch

from check_atb_mpi_invariance import author_imports, digest
from prepare_b36_rule_b import ROOT, now, read, save

CFG=ROOT/'configs/any2bokeh-subject-coc-statistics-20261021.json'


def write_csv(path,rows):
    assert not path.exists(),path
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def masks_for(label,label_id):
    native=(label!=0) if label_id=='union' else (label==int(label_id))
    display=cv2.resize(native.astype(np.uint8),(1024,576),interpolation=cv2.INTER_NEAREST)>0
    inside=cv2.erode(display.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    ni=native & (cv2.resize(inside.astype(np.uint8),(label.shape[1],label.shape[0]),interpolation=cv2.INTER_NEAREST)>0)
    return {'native':dict(whole=native,interior=ni,boundary=native&~ni),
            'actual_condition':dict(whole=display,interior=inside,boundary=display&~inside)}


def values(a,mask):
    x=a[mask];p=int(x.size)
    return dict(pixels=p,median=None if not p else float(np.median(x)),p90=None if not p else float(np.quantile(x,.9)),
        p95=None if not p else float(np.quantile(x,.95)),
        **{f'gt{str(t).replace(".","_")}':None if not p else float(np.mean(x>t)) for t in [.5,1,2,4]})


def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);c=read(CFG);out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    assert not (out/'statistics-complete.json').exists()
    split=read(ROOT/c['split_record'])['rows'];explore=sorted(r['scene_id'] for r in split if r['dataset']=='DAVIS' and r['split']=='exploration')
    assert len(explore)==35 and all(s in explore for s in ['rhino','stroller','blackswan','cows','mallard-water'])
    labels_by_scene={};exclusions={};bid={}
    for file in c['B_manifests']:
        for r in read(ROOT/file)['rows']:
            if r['dataset']=='DAVIS':
                bid[r['scene_id']]=r['sample_id'];exclusions.setdefault(r['scene_id'],[]).append(file)
    for s in read(ROOT/'configs/davis-baselines.json')['preflight_sequences']:exclusions.setdefault(s,[]).append('six-method fixed8')
    root=out.parent
    for p in root.rglob('*.html'):
        if out in p.parents or 'browser-check' in p.parts:continue
        for s in set(re.findall(r'"scene_id"\s*:\s*"([a-z][a-z0-9-]+)"',p.read_text(errors='replace'))):
            exclusions.setdefault(s,[]).append(str(p.relative_to(ROOT)))
    # Save exclusion inventory before ranking or prediction access.
    plan=out/'statistics-plan.json'
    if plan.exists():
        assert read(plan)['config']==c and read(plan)['exploration']==explore
        assert read(plan)['excluded_sources']==exclusions
    else:
        save(plan,dict(saved_UTC=now(),config=c,exploration=explore,
            excluded_sources=exclusions,source_overlap_with_holdout=False,model_predictions_opened=False))
    _,_,_,_,Dataset,_=author_imports()
    all_rows=[];scene_info=[];checks=[]
    for scene in explore:
        local=out/'CPU-statistics'/scene;local.mkdir(parents=True,exist_ok=True)
        framefile=local/'frames.json'
        if framefile.exists():
            done=read(framefile)
            with Image.open(ROOT/'data/raw/DAVIS-2017/DAVIS/Annotations/Full-Resolution'/scene/'00000.png') as im:done['scene_info']['native_size']=list(im.size)
            all_rows.extend(done['rows']);scene_info.append(done['scene_info']);checks.append(done['loader_checks']);print('REUSE_CPU',scene,flush=True);continue
        folder=ROOT/'data/processed/davis2016-v2/eval'/scene
        geo=read(ROOT/'data/processed/davis2016-v2/sequences'/scene/'geometry.json')
        labels=[np.asarray(Image.open(ROOT/'data/raw/DAVIS-2017/DAVIS/Annotations/Full-Resolution'/scene/f'{t:05d}.png')) for t in range(25)]
        assert len({a.shape for a in labels})==1
        ids=sorted(set(int(v) for a in labels for v in np.unique(a) if v!=0));rowids=['union']+list(map(str,ids))
        info=dict(scene_id=scene,sample_id=bid.get(scene,''),source_group='DAVIS:'+scene,IDs=ids,
            instance_count=len(ids),instance_type='multi-instance' if len(ids)>1 else 'single-instance',native_size=[labels[0].shape[1],labels[0].shape[0]])
        rows=[]
        for t in range(25):
            f=geo['frames'][t];q=np.asarray(Image.open(folder/'disparity'/f['file']));assert q.shape==labels[t].shape
            assert float(Path(f['file']).stem.split('_zf_')[1])==f['focus']
            assert np.array_equal(np.asarray(Image.open(folder/'mask'/f'{t:05d}.png'))!=0,labels[t]!=0)
            raw=16*np.abs(q.astype(np.float64)/255-f['focus'])
            for label_id in rowids:
                for region,m in masks_for(labels[t],label_id)['native'].items():
                    rows.append(dict(scene_id=scene,sample_id=bid.get(scene,''),frame_id=t,grid='native',label_id=label_id,region=region,
                        focus=f['focus'],K=16,instance_count=len(ids),**values(raw,m)))
        # Existing frozen input CSV; no prediction file is read here.
        path=ROOT/'outputs/davis-baselines-20260929/any2bokeh/audit'/scene/'input.csv'
        ds=Dataset(str(path),sample_size=(576,1024));assert float(ds.K[0])==16
        norm=ds.pixel_transforms.transforms[-1];calls=0;seen={};frame_seen=set();groupids=[list(range(s,min(s+8,25))) for s in [0,4,8,12,16,20]]
        def before_normalize(module,args):
            nonlocal calls
            calls+=1
            if calls%2: return  # odd=RGB, even=CoC in the pinned author get_batch
            group=(calls//2)-1;data=args[0]
            assert tuple(data.shape)==(len(groupids[group]),3,576,1024)
            for j,t in enumerate(groupids[group]):
                assert torch.equal(data[j,0],data[j,1]) and torch.equal(data[j,0],data[j,2])
                sha=digest(data[j])
                if t in seen:assert sha==seen[t];continue
                seen[t]=sha;frame_seen.add(t);actual=data[j,0].numpy()*16
                assert np.isfinite(actual).all() and actual.min()>=0
                for label_id in rowids:
                    for region,m in masks_for(labels[t],label_id)['actual_condition'].items():
                        rows.append(dict(scene_id=scene,sample_id=bid.get(scene,''),frame_id=t,grid='actual_condition',label_id=label_id,region=region,
                            focus=geo['frames'][t]['focus'],K=16,instance_count=len(ids),**values(actual,m)))
        hook=norm.register_forward_pre_hook(before_normalize)
        try:batch=ds[0]
        finally:hook.remove()
        assert calls==12 and frame_seen==set(range(25)) and len(batch)==6
        check=dict(scene_id=scene,Normalize_calls=12,CoC_source_frames=25,real_frame_slots=45,
            all_preNormalize_overlap_copies_bitwise_equal=True,actual_author_transform_observed=True,
            preNormalize_sha256={str(k):v for k,v in seen.items()})
        del batch,ds
        done=dict(rows=rows,scene_info=info,loader_checks=check)
        save(framefile,done);all_rows.extend(rows);scene_info.append(info);checks.append(check)
        print('CPU_DONE',scene,'IDs',ids,'rows',len(rows),flush=True)
    metrics=['median','p90','p95','gt0_5','gt1','gt2','gt4'];summary=[]
    for info in scene_info:
        for grid in ['native','actual_condition']:
            for label_id in ['union']+list(map(str,info['IDs'])):
                for region in ['whole','interior','boundary']:
                    group=[r for r in all_rows if r['scene_id']==info['scene_id'] and r['grid']==grid and r['label_id']==label_id and r['region']==region]
                    row={k:info[k] for k in ['scene_id','sample_id','instance_count','instance_type']};row.update(grid=grid,label_id=label_id,region=region,frames=25)
                    assert len(group)==25
                    for metric in metrics:
                        valid=[r for r in group if r[metric] is not None]
                        peak=max(valid,key=lambda r:(r[metric],-r['frame_id'])) if valid else None
                        row[metric+'_clip_median']=float(np.median([r[metric] for r in valid])) if valid else None
                        row[metric+'_valid_frames']=len(valid);row[metric+'_highest']=None if peak is None else peak[metric];row[metric+'_highest_frame']=None if peak is None else peak['frame_id']
                    summary.append(row)
    rankings=[]
    for grid in ['actual_condition','native']:
        source=sorted([r for r in summary if r['grid']==grid and r['label_id']=='union' and r['region']=='interior'],key=lambda r:(-(r['gt1_clip_median'] if r['gt1_clip_median'] is not None else -1),r['scene_id']))
        for rank,r in enumerate(source,1):rankings.append(dict(rank=rank,grid=grid,scene_id=r['scene_id'],sample_id=r['sample_id'],instance_type=r['instance_type'],instance_count=r['instance_count'],union_interior_gt1_clip_median=r['gt1_clip_median'],highest_fraction=r['gt1_highest'],highest_frame=r['gt1_highest_frame'],user_report=c['known_user_labels'].get(r['scene_id'],''),excluded_from_statistical_selection=r['scene_id'] in exclusions))
    primary=[r for r in rankings if r['grid']=='actual_condition'];chosen=next(r for r in primary if not r['excluded_from_statistical_selection'])
    selection=dict(saved_UTC=now(),rule=c['primary_ranking'],chosen=chosen,excluded_sources=exclusions,
        eligible_in_order=[r['scene_id'] for r in primary if not r['excluded_from_statistical_selection']],
        model_predictions_opened_before_selection=False,exploration_only=True,blind_test=False)
    prefix=c['record_prefix'];write_csv(ROOT/(prefix+'-frames.csv'),all_rows);write_csv(ROOT/(prefix+'-scenes.csv'),summary);write_csv(ROOT/(prefix+'-ranking.csv'),rankings)
    save(ROOT/(prefix+'-selection.json'),selection)
    result=dict(status='35段探索组双口径CPU输入统计完成；选样先于新输出查看',config=c,created_UTC=now(),scenes=scene_info,
        primary_ranking=primary,native_ranking=[r for r in rankings if r['grid']=='native'],loader_checks=checks,selection_record=prefix+'-selection.json',
        valid_sequences=35,valid_source_frames=875,heldout_video_inputs_opened=0,neural_inference_runs=0,
        annotation_label_scope='All original nonzero IDs, including255 if present; ID counts are not disjoint connected-component counts.',
        limitations=['About6 user-labelled examples: descriptions/ranks only, no correlation or significance.',
            'Large subject defocus is not depth error. Whole-subject ranking can miss a small head region; mismatched order is retained.',
            'Primary actual condition and native math grids are separate; no ordering expected as a pass criterion.'])
    save(ROOT/(prefix+'.json'),result);save(out/'statistics-complete.json',dict(created_UTC=now(),record=prefix+'.json',selected_scene=chosen['scene_id']))
    print('SELECTED',chosen,flush=True)


if __name__=='__main__':main()
