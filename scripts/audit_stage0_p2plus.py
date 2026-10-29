"""Controlled audit of scalar depth sampling and focus-query readout.

Rules consume the actual P2+ feature tensor and focus mask where advertised.
No backbone inference or new training is used by train-audit / diagnostic.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.focus_relative_probe import read_json,save_json,sha,make_focus_masks,focus_boxes,confusion,metrics,rule_score
from scripts.run_v2_stage0 import scene_features,write_csv
AUDIT=ROOT/'configs/v2-stage0-p2plus-audit-20261029.json'
VARIANTS=['dense_P2','sampled_dense_focus','sampled_query_fixed','sampled_matched_rule','dense_mean_rule']


def input_scores(depth,masks):
    """Only dense_P2 and sampled_dense_focus use the extra dense median focus."""
    d=torch.as_tensor(np.asarray(depth).copy(),dtype=torch.float32)
    log=torch.log(d);center=log.flatten(1).median(1).values[:,None,None]
    full=log-center
    coarse=F.interpolate(full[:,None],size=(16,37),mode='nearest-exact')[:,0]
    weights=F.interpolate(masks[:,None],size=(16,37),mode='area')[:,0]
    denom=weights.sum((-1,-2))
    query=(coarse[:,None]*weights[None]).sum((-1,-2))/denom[None]
    dense_mean=(full[:,None]*masks[None]).sum((-1,-2))/masks.sum((-1,-2))[None]
    qmedian=[]
    for mask in masks:
        qmedian.append(torch.log(d[:,mask.bool()].median(1).values)-center[:,0,0])
    # numpy median matches the original even-count median, unlike torch.median.
    for fi,mask in enumerate(masks):
        qmedian[fi]=torch.log(torch.from_numpy(np.median(depth[:,mask.numpy().astype(bool)],axis=1)))-center[:,0,0]
    median=torch.stack(qmedian,1)
    def up(q):
        low=q[:,:,None,None]-coarse[:,None]
        return F.interpolate(low.flatten(0,1)[:,None],size=depth.shape[-2:],mode='bilinear',align_corners=False)[:,0].reshape(depth.shape[0],masks.shape[0],*depth.shape[-2:]).numpy()
    dense_scores=np.empty((depth.shape[0],len(masks),*depth.shape[-2:]),np.float32)
    for t in range(depth.shape[0]):
        for fi,mask in enumerate(masks):
            yy,xx=torch.where(mask.bool());box=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]
            dense_scores[t,fi]=rule_score(depth[t],box,True)
    sampled=up(query)
    scores=dict(dense_P2=dense_scores,sampled_dense_focus=up(median),sampled_query_fixed=sampled,
                sampled_matched_rule=sampled,dense_mean_rule=(dense_mean[:,:,None,None]-full[:,None]).numpy())
    reconstructed=F.interpolate(coarse[:,None],size=depth.shape[-2:],mode='bilinear',align_corners=False)[:,0]
    center_mask=F.interpolate(masks[:,None],size=(16,37),mode='nearest-exact')[:,0]
    outside=((weights*(1-center_mask)).sum((-1,-2))/denom).numpy()
    info=dict(coarse=coarse.numpy(),query=query.numpy(),median=median.numpy(),dense_mean=dense_mean.numpy(),
              reconstruction_error=(reconstructed-full).abs().numpy(),outside_focus_center_weight_fraction=outside.tolist(),weights=weights.numpy())
    return scores,info


def quantile_stats(a):
    a=np.asarray(a)
    return dict(mean=float(a.mean()),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),maximum=float(a.max())) if a.size else dict.fromkeys(['mean','p50','p90','maximum'])


def rows_for(name,split,labels,boundary,scores,thresholds):
    rows=[]
    for method,score in scores.items():
        threshold=thresholds[method]
        pred=np.ones(score.shape,np.uint8);pred[score>threshold]=0;pred[score < -threshold]=2
        for region in ['all_valid','boundary','interior']:
            m=None if region=='all_valid' else np.broadcast_to((boundary if region=='boundary' else ~boundary)[:,None],labels.shape)
            rows.append(dict(method=method,sequence=name,split=split,region=region,threshold=threshold,**metrics(confusion(labels,pred,m))))
    return rows


def hist_add(hist,total,score,label,thresholds):
    m=label<3;s=score[m];y=label[m]
    total+=np.bincount(y,minlength=3)
    for sign,select in enumerate([s>0,s<0]):
        bins=np.searchsorted(thresholds,np.abs(s[select]),side='left')
        hist[:,sign]+=np.bincount(y[select]*(len(thresholds)+1)+bins,minlength=3*(len(thresholds)+1)).reshape(3,-1)


def calibration(histograms,thresholds):
    curves=[]
    for i,t in enumerate(thresholds):
        per_source={}
        for name,(hist,total) in histograms.items():
            tail=np.cumsum(hist[:,:,::-1],axis=2)[:,:,::-1]
            cm=np.zeros((3,3),np.int64);cm[:,0]=tail[:,0,i+1];cm[:,2]=tail[:,1,i+1];cm[:,1]=total-cm[:,0]-cm[:,2]
            per_source[name]=metrics(cm)
        curves.append(dict(threshold=float(t),Macro_F1=float(np.mean([r['Macro_F1'] for r in per_source.values()])),per_source=per_source))
    chosen=max(curves,key=lambda r:(r['Macro_F1'],-r['threshold']))
    return dict(threshold=chosen['threshold'],source_equal_training_Macro_F1=chosen['Macro_F1'],per_source=chosen['per_source'],
                training_curve=[{k:v for k,v in r.items() if k!='per_source'} for r in curves],no_development_fitting=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['train-audit','diagnostic']);args=parser.parse_args()
    torch.set_num_threads(6)
    audit=read_json(AUDIT);c=read_json(ROOT/audit['base_config']);base=ROOT/c['output'];out=ROOT/audit['output'];out.mkdir(exist_ok=True)
    masks=torch.from_numpy(make_focus_masks(c));ts=np.array(audit['threshold_candidates']);input_rows=[];measurements=[]
    sources=[s for s in c['sequences'] if args.phase=='diagnostic' or s['split']=='pilot_train']
    histograms={m:{} for m in ['sampled_matched_rule','dense_mean_rule']};outside=None
    summaries=read_json(ROOT/(c['record_prefix']+'-results.json'))
    for scene in sources:
        name=scene['sequence'];data=np.load(base/'data'/f'{name}.npz');depth=np.load(base/'features'/name/'depth.npy')
        scores,info=input_scores(depth,masks)
        actual=scene_features(c,base,name,'P2plus')[:,0].numpy()
        assert np.array_equal(actual,info['coarse']), 'Must audit actual P2+ information'
        outside=info['outside_focus_center_weight_fraction']
        if args.phase=='train-audit':
            for method in histograms:
                h=np.zeros((3,2,len(ts)+1),np.int64);total=np.zeros(3,np.int64)
                hist_add(h,total,scores[method],data['labels'],ts);histograms[method][name]=(h,total)
        else:
            calibration_record=read_json(out/'train-audit.json')['calibrations']
            thresholds={m:.13 for m in VARIANTS};thresholds.update({m:calibration_record[m]['threshold'] for m in histograms})
            rows=rows_for(name,scene['split'],data['labels'],data['boundary'],scores,thresholds);measurements.extend(rows)
            baseline=next(r for r in summaries['per_source'] if r['sequence']==name and r['method']=='P2' and r['region']=='all_valid')
            assert next(r['confusion'] for r in rows if r['method']=='dense_P2' and r['region']=='all_valid')==baseline['confusion']
        dense=scores['dense_P2'];sampled=scores['sampled_dense_focus'];query_score=scores['sampled_query_fixed']
        classify=lambda a:np.where(a>.13,0,np.where(a < -.13,2,1))
        for t in range(25):
            for region in ['all_valid','boundary','interior']:
                m=data['valid'][t] if region=='all_valid' else data['valid'][t]&(data['boundary'][t] if region=='boundary' else ~data['boundary'][t])
                input_rows.append(dict(sequence=name,split=scene['split'],frame_id=t+1,region=region,pixels=int(m.sum()),
                    **{'log_depth_reconstruction_'+k:v for k,v in quantile_stats(info['reconstruction_error'][t][m]).items()},
                    dense_vs_sampled_fixed_focus_class_disagreement=float((classify(dense[t])[:,m]!=classify(sampled[t])[:,m]).mean()),
                    dense_vs_sampled_actual_query_class_disagreement=float((classify(dense[t])[:,m]!=classify(query_score[t])[:,m]).mean())))
            for fi in range(5):
                input_rows.append(dict(sequence=name,split=scene['split'],frame_id=t+1,region='focus_reference',focus_id=fi,
                     query_minus_dense_log_median=float(info['query'][t,fi]-info['median'][t,fi]),
                     query_minus_dense_log_mean=float(info['query'][t,fi]-info['dense_mean'][t,fi]),
                     dense_log_mean_minus_log_median=float(info['dense_mean'][t,fi]-info['median'][t,fi]),
                     focus_weight_on_centers_outside_focus=outside[fi]))
        print('AUDITED',args.phase,name,flush=True)
    if args.phase=='train-audit':
        cal={m:calibration(h,ts) for m,h in histograms.items()}
        reference_rows=[r for r in input_rows if r['region']=='focus_reference']
        regions=[r for r in input_rows if r['region']=='all_valid']
        result=dict(config_sha256=sha(AUDIT),training_sources=[s['sequence'] for s in sources],calibrations=cal,
             outside_focus_center_weights=outside,
             mean_spatial_only_class_disagreement=float(np.mean([r['dense_vs_sampled_fixed_focus_class_disagreement'] for r in regions])),
             mean_including_query_class_disagreement=float(np.mean([r['dense_vs_sampled_actual_query_class_disagreement'] for r in regions])),
             query_abs_error_vs_dense_median=quantile_stats([abs(r['query_minus_dense_log_median']) for r in reference_rows]),
             frozen_candidate_gate_passed=bool(max(outside)>0 and np.mean([r['dense_vs_sampled_actual_query_class_disagreement'] for r in regions])>.01),
             candidate='dense spatial readout only; no architecture/loss/training-budget change',dev_results_opened_for_candidate_selection=False)
        save_json(out/'train-audit.json',result);save_json(ROOT/(audit['record_prefix']+'-train-audit.json'),result)
        write_csv(ROOT/(audit['record_prefix']+'-train-inputs.csv'),input_rows)
        print('TRAIN_AUDIT',json.dumps(result,ensure_ascii=False)[:1800],flush=True)
    else:
        for old_method in ['P2plus','P3']:
            measurements.extend(r for r in summaries['per_source'] if r['method']==old_method)
        result=dict(config_sha256=sha(AUDIT),per_source=measurements,source_split='original six already-used sources; diagnostic only',
                    outside_focus_center_weights=outside,actual_P2plus_feature_bitwise_verified=True,
                    dense_rule_original_confusions_verified=True,no_new_backbone_or_training=True)
        save_json(out/'diagnostic.json',result);save_json(ROOT/(audit['record_prefix']+'-diagnostic.json'),result)
        write_csv(ROOT/(audit['record_prefix']+'-diagnostic.csv'),measurements)
        write_csv(ROOT/(audit['record_prefix']+'-inputs.csv'),input_rows)
        for method in VARIANTS+['P2plus','P3']:
            for reg in ['all_valid','boundary']:
                rs=[r for r in measurements if r['method']==method and r['split']=='pilot_dev' and r['region']==reg]
                print('DEV_DIAGNOSTIC',method,reg,float(np.mean([r['Macro_F1'] for r in rs])),flush=True)


if __name__=='__main__':main()
