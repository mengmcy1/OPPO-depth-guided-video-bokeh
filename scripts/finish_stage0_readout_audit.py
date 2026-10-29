"""Finish the frozen P2+ grid intervention; never train on new sources.

Reuse original labels/weights on the six-source diagnostic. New-source labels
use the identical native focus/reference and nearest-pixel protocol, without
requiring a VDA prediction or creating a fabricated VDA input.
"""
import argparse
import copy
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.focus_relative_probe import (read_json, save_json, sha, depth_read,
    coordinate_indices, focus_boxes, make_focus_masks, confusion, metrics, RelationProbe)
from scripts import run_v2_stage0 as run
from scripts.audit_stage0_p2plus import input_scores

BASE_PATH = ROOT/'configs/v2-stage0-20261029.json'
DENSE_PATH = ROOT/'configs/v2-stage0-p2plus-dense-grid-20261029.json'
NEW_PATH = ROOT/'configs/v2-stage0-p2plus-audit-independent-20261029.json'
BASE = read_json(BASE_PATH)
DENSE = read_json(DENSE_PATH)
NEW = read_json(NEW_PATH)
OUT = ROOT/'outputs/v2-stage0-p2plus-audit-20261029'
PREFIX = ROOT/'03_实验记录/records/v2-stage0-p2plus-audit-20261029'


def labels_for_scene(c, scene):
    # Same operations and precision as the original build_labels implementation.
    w,h = c['preprocessing']['model_wh']; nw,nh = c['preprocessing']['native_wh']
    xi,yi = coordinate_indices([nw,nh],[w,h])
    labels=[]; validities=[]; edges=[]; depths=[]; rows=[]
    for t,frame in enumerate(scene['frames']):
        folder=ROOT/c['raw_root']/'training'
        z=depth_read(folder/'depth'/scene['sequence']/f'frame_{frame:04d}.dpt')
        assert z.shape==(nh,nw)
        rgb=folder/'clean'/scene['sequence']/f'frame_{frame:04d}.png'
        with Image.open(rgb) as im: assert im.size==(nw,nh)
        valid=np.isfinite(z)&(z>0); small=z[yi[:,None],xi[None,:]]
        vm=valid[yi[:,None],xi[None,:]]
        logz=np.log(np.where(vm,small,1)).astype(np.float32)
        complete=cv2.erode(vm.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        edge=complete & ((cv2.dilate(logz,np.ones((3,3),np.uint8))-cv2.erode(logz,np.ones((3,3),np.uint8)))>c['boundary']['threshold'])
        edge=(cv2.dilate(edge.astype(np.uint8),np.ones((3,3),np.uint8))>0)&vm
        current=[]
        for fi,(fx,fy,fw,fh) in enumerate(focus_boxes(c)):
            a,b=int(np.floor(fx*nw/w)),int(np.floor(fy*nh/h))
            right,bottom=int(np.ceil((fx+fw)*nw/w)),int(np.ceil((fy+fh)*nh/h))
            native=valid[b:bottom,a:right]; display=vm[fy:fy+fh,fx:fx+fw]
            ok=display.sum()>=c['focus']['minimum_valid_pixels'] and display.mean()>=c['focus']['minimum_valid_fraction'] and native.any()
            ref=float(np.median(z[b:bottom,a:right][native])) if ok else None
            lab=np.full((h,w),255,np.uint8)
            if ok:
                r=np.log(ref/np.where(vm,small,1));eps=c['labels']['epsilon']
                lab[vm]=1;lab[vm&(r>eps)]=0;lab[vm&(r < -eps)]=2
            current.append(lab)
            rows.append(dict(sequence=scene['sequence'],source_family=scene['source_family'],split=scene['split'],frame_id=frame,
                focus_id=fi,valid_problem=bool(ok),GT_Z_focus=ref,display_xywh=[fx,fy,fw,fh],native_xyxy=[a,b,right,bottom],
                label_counts=[int((lab==j).sum()) for j in range(3)],valid_pixels=int(vm.sum()),
                RGB=str(rgb.relative_to(ROOT))))
        labels.append(np.stack(current));validities.append(vm);edges.append(edge);depths.append(small)
    return dict(labels=np.stack(labels),valid=np.stack(validities),boundary=np.stack(edges),Z=np.stack(depths)),rows


def new_config():
    c=copy.deepcopy(BASE)
    c.update(raw_root=NEW['raw_root'],sequences=NEW['sequences'],output=NEW['output'])
    return c


def prepare_new():
    run.setup(BASE)
    out=ROOT/NEW['output'];(out/'data').mkdir(parents=True,exist_ok=True)
    if (out/'labels-check.json').exists():
        print('REUSE_NEW_LABELS');return
    # Rebuild all old labels before applying exactly this protocol to new data.
    matches=[]
    for s in BASE['sequences']:
        data,_=labels_for_scene(BASE,s)
        with np.load(ROOT/BASE['output']/'data'/f"{s['sequence']}.npz") as old:
            for key,value in data.items(): assert np.array_equal(value,old[key]),(s['sequence'],key)
        matches.append(s['sequence'])
    old_families={s['source_family'] for s in BASE['sequences']}
    assert not old_families & {s['source_family'] for s in NEW['sequences']}
    c=new_config();rows=[]
    for s in c['sequences']:
        data,r=labels_for_scene(c,s);dest=out/'data'/f"{s['sequence']}.npz"
        assert not dest.exists();np.savez_compressed(dest,**data);rows.extend(r)
        save_json(out/'data'/f"{s['sequence']}-manifest.json",r)
    record=dict(original_label_rebuild_bitwise_sources=matches,new_sources=NEW['sequences'],source_family_overlap=0,
        questions=len(rows),valid_questions=sum(r['valid_problem'] for r in rows),
        independent_config_sha256=sha(NEW_PATH),no_VDA_input_needed=True,no_model_fitting=True,
        validity='same raw finite-positive rule, includes distant sentinel depths; no transparent/mixed ground truth mask',
        questions_manifest=rows)
    save_json(out/'labels-check.json',record)
    save_json(str(PREFIX)+'-independent-labels.json',record)
    print('PREPARED_NEW',len(rows),flush=True)


def summarize(rows):
    result=[]
    for split in sorted({r['split'] for r in rows}):
        for method in sorted({r['method'] for r in rows}):
            for region in ['all_valid','boundary','interior']:
                rs=[r for r in rows if (r['split'],r['method'],r['region'])==(split,method,region)]
                if rs: result.append(dict(method=method,split=split,region=region,sources=len(rs),
                    Macro_F1=float(np.mean([r['Macro_F1'] for r in rs])),
                    Balanced_Accuracy=float(np.mean([r['Balanced_Accuracy'] for r in rs]))))
    return result


def metric_rows(pred,data,s,method):
    lab=data['labels'];boundary=data['boundary'];rows=[]
    for region in ['all_valid','boundary','interior']:
        mask=None if region=='all_valid' else np.broadcast_to((boundary if region=='boundary' else ~boundary)[:,None],lab.shape)
        rows.append(dict(method=method,sequence=s['sequence'],source_family=s['source_family'],split=s['split'],region=region,
            **metrics(confusion(lab,pred,mask))))
    return rows


def evaluate(independent=False):
    run.setup(BASE,True)
    tag='independent' if independent else 'original'
    dest=OUT/tag;dest.mkdir(exist_ok=True)
    if (dest/'evaluation.json').exists():print('REUSE_EVALUATION',tag);return
    c=new_config() if independent else BASE
    data_out=ROOT/c['output'];masks=torch.from_numpy(make_focus_masks(c)).unsqueeze(1).cuda()
    specs={'P2plus_dense_grid':(DENSE,DENSE_PATH,'P2plus')}
    if independent: specs.update(P2plus_original=(BASE,BASE_PATH,'P2plus'),P3_frozen=(BASE,BASE_PATH,'P3'))
    models={};checkpoints={}
    for name,(mc,path,method) in specs.items():
        cp=ROOT/mc['output']/'heads'/method/'final.pt'
        ckpt=torch.load(cp,map_location='cuda',weights_only=True)
        assert ckpt['step']==1000 and ckpt['config_sha256']==sha(path)
        model=RelationProbe(2048 if method=='P3' else 1,mc).cuda().eval()
        model.load_state_dict(ckpt['model'],strict=True);models[name]=model
        checkpoints[name]=sha(cp)
    rows=[];checks=[];runtime=[]
    if not independent:
        old=read_json(OUT/'diagnostic.json')['per_source']
        rows.extend(old)
    for s in c['sequences']:
        scene=s['sequence'];data=np.load(data_out/'data'/f'{scene}.npz')
        pd=dest/'predictions'/scene;pd.mkdir(parents=True,exist_ok=True)
        if independent:
            scores,_=input_scores(np.load(data_out/'features'/scene/'depth.npy'),masks[:,0].cpu())
            for name in ['dense_P2','sampled_matched_rule','dense_mean_rule']:
                score=scores[name];t=NEW['thresholds'][name]
                pred=np.where(score>t,0,np.where(score < -t,2,1)).astype(np.uint8)
                np.save(pd/f'{name}.npy',pred);rows.extend(metric_rows(pred,data,s,name))
            del scores
        for name,model in models.items():
            mc,path,method=specs[name]
            features=run.scene_features(mc,data_out,scene,method).cuda()
            pred=np.empty_like(data['labels']);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.monotonic()
            with torch.inference_mode():
                for t in range(len(features)):
                    logits=model(features[t:t+1].expand(5,-1,-1,-1),masks)
                    assert torch.isfinite(logits).all()
                    pred[t]=logits.argmax(1).cpu().numpy().astype(np.uint8)
                torch.cuda.synchronize()
                runtime.append(dict(method=name,sequence=scene,seconds=time.monotonic()-t0,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
                # Matched batch/precision fresh load, compare saved predictions.
                clone=RelationProbe(model.channels,mc).cuda().eval()
                clone.load_state_dict(torch.load(ROOT/mc['output']/'heads'/method/'final.pt',map_location='cuda',weights_only=True)['model'])
                for t in [0,len(features)-1]:
                    replay=clone(features[t:t+1].expand(5,-1,-1,-1),masks).argmax(1).cpu().numpy().astype(np.uint8)
                    assert np.array_equal(replay,pred[t]),(name,scene,t)
                del clone
            np.save(pd/f'{name}.npy',pred);rows.extend(metric_rows(pred,data,s,name))
            checks.append(dict(method=name,sequence=scene,reloaded_first_last_predictions_bitwise=True))
            del features
        print('EVALUATED_AUDIT',tag,scene,flush=True)
    summary=summarize(rows)
    record=dict(tag=tag,per_source=rows,summary=summary,reload_checks=checks,checkpoints=checkpoints,runtime=runtime,
        base_config_sha256=sha(BASE_PATH),dense_config_sha256=sha(DENSE_PATH),independent_config_sha256=sha(NEW_PATH),
        no_new_threshold_selection=True,no_training_on_new_sources=True,not_Bokeh_quality=True)
    if independent:
        def metric(method,region):return next(r['Macro_F1'] for r in summary if r['method']==method and r['region']==region)
        delta=metric('P2plus_dense_grid','all_valid')-metric('P2plus_original','all_valid')
        bd=metric('P2plus_dense_grid','boundary')-metric('P2plus_original','boundary')
        record['prespecified_decision']=dict(dense_minus_original=delta,boundary_delta=bd,
            screen_passed=bool(delta>=NEW['engineering_delta_min'] and bd>=NEW['boundary_delta_min']),
            P3_minus_dense=metric('P3_frozen','all_valid')-metric('P2plus_dense_grid','all_valid'))
    save_json(dest/'evaluation.json',record);save_json(str(PREFIX)+f'-{tag}-results.json',record)
    run.write_csv(str(PREFIX)+f'-{tag}-per-source.csv',rows)
    run.write_csv(str(PREFIX)+f'-{tag}-summary.csv',summary)
    for r in summary:
        if r['region']=='all_valid': print('SUMMARY',tag,r,flush=True)


def report():
    dest=OUT/'report';dest.mkdir(exist_ok=True)
    palette=np.array([[237,105,65],[100,194,108],[77,128,222],[160,160,160]],np.uint8)
    links=[]
    for tag,c in [('original',BASE),('independent',new_config())]:
        if not (OUT/tag/'evaluation.json').exists():continue
        for s in c['sequences']:
            scene=s['sequence'];data=np.load(ROOT/c['output']/'data'/f'{scene}.npz')
            predictions={}
            if tag=='original':
                for name,alias in [('P2','dense_P2'),('P2plus','P2plus_original'),('P3','P3_frozen')]:
                    predictions[alias]=np.load(ROOT/BASE['output']/'predictions'/scene/f'{name}.npy')
            else:
                for name in ['dense_P2','P2plus_original','P3_frozen']:
                    predictions[name]=np.load(OUT/tag/'predictions'/scene/f'{name}.npy')
            predictions['P2plus_dense_grid']=np.load(OUT/tag/'predictions'/scene/'P2plus_dense_grid.npy')
            order=['dense_P2','P2plus_original','P2plus_dense_grid','P3_frozen']
            sheet=Image.new('RGB',(518*3,254*6),'white');draw=ImageDraw.Draw(sheet)
            for ri,t in enumerate([0,12,24]):
                rgb=Image.open(ROOT/c['raw_root']/'training/clean'/scene/f'frame_{t+1:04d}.png').convert('RGB').resize((518,224),Image.Resampling.BICUBIC)
                x,y,w,h=focus_boxes(c)[1];ImageDraw.Draw(rgb).rectangle([x,y,x+w-1,y+h-1],outline='yellow',width=2)
                lab=data['labels'][t,1];images=[rgb,Image.fromarray(palette[np.where(lab<3,lab,3)])]
                images.extend(Image.fromarray(palette[predictions[m][t,1]]) for m in order)
                for j,im in enumerate(images):
                    xx=(j%3)*518;yy=(ri*2+j//3)*254
                    sheet.paste(im,(xx,yy+30));draw.text((xx+4,yy+7),f'{scene} f{t+1} '+(['RGB/focus1','GT']+order)[j],fill='black')
            path=dest/f'{scene}.png';sheet.save(path);links.append((scene,path.name))
    html='<meta charset="utf-8"><title>P2+ readout audit</title><h1>P2+ 空间读出核查</h1><p>固定源帧1/13/25、焦点1。橙=近，绿=相近，蓝=远，灰=无效。这是几何分类，不是虚化图。所有来源保留，不按表现筛选。</p>'
    for scene,path in links:html+=f'<h2>{scene}</h2><a href="{path}">原尺寸图</a><br><img src="{path}" style="max-width:100%">'
    (dest/'index.html').write_text(html)
    print('REPORT',len(links))


def verify():
    import hashlib
    import zlib
    run.setup(BASE)
    old_train=read_json(ROOT/BASE['output']/'heads/P2plus/training.json')
    dense_train=read_json(ROOT/DENSE['output']/'heads/P2plus/training.json')
    unchanged=['steps','frame_focus_exposures','training_schedule_sha256','initial_relation_sha256',
               'trainable_parameters','projection_parameters','relation_parameters']
    for k in unchanged:assert old_train[k]==dense_train[k],k
    for k in BASE:
        if k not in ['output','record_prefix']:assert BASE[k]==DENSE[k],k
    assert DENSE['P2plus_spatial_grid']==[224,518]
    checkpoint=torch.load(ROOT/DENSE['output']/'heads/P2plus/final.pt',map_location='cpu',weights_only=True)
    assert checkpoint['config_sha256']==sha(DENSE_PATH) and checkpoint['step']==1000
    assert all(int(s['step'])==1000 for s in checkpoint['optimizer']['state'].values())
    assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
    family_order=sorted(NEW['candidate_family_order'],key=lambda n:hashlib.sha256(('20261029:P2plus-audit-independent:'+n).encode()).hexdigest())
    assert family_order==NEW['candidate_family_order']
    assert [s['source_family'] for s in NEW['sequences']]==family_order[:2]
    # Hash ordering alone is checked here; original metadata selections remain
    # explicit in the frozen config, not changed by evaluation results.
    raw=ROOT/NEW['raw_root'];index=read_json(raw/'selected-members.json')
    for path,entry in index.items():
        b=(raw/path).read_bytes();assert len(b)==entry['bytes'] and zlib.crc32(b)==entry['crc32']
    all_rows=[];arrays=0;reloads=0
    for tag,c in [('original',BASE),('independent',new_config())]:
        result=read_json(OUT/tag/'evaluation.json');all_rows.extend(result['per_source'])
        reloads+=len(result['reload_checks'])*2
        for s in c['sequences']:
            scene=s['sequence'];data=np.load(ROOT/c['output']/'data'/f'{scene}.npz')
            for p in (OUT/tag/'predictions'/scene).glob('*.npy'):
                pred=np.load(p);assert pred.shape==data['labels'].shape and pred.dtype==np.uint8 and pred.max()<=2
                expected=metric_rows(pred,data,s,p.stem)
                for r in expected:
                    recorded=next(x for x in result['per_source'] if (x['method'],x['sequence'],x['region'])==(r['method'],scene,r['region']))
                    assert r['confusion']==recorded['confusion']
                arrays+=1
        recomputed=summarize(result['per_source']);assert recomputed==result['summary']
    decision=read_json(OUT/'independent/evaluation.json')['prespecified_decision']
    check=dict(status='PASS',training_unchanged_keys=unchanged,only_config_intervention='P2plus spatial grid',
        parameter_count=dense_train['trainable_parameters'],optimizer_step=1000,checkpoint_finite=True,
        raw_members_CRC_verified=len(index),metadata_family_hash_order=family_order,
        new_prediction_arrays_CM_recomputed=arrays,matched_batch_first_last_frame_reloads=reloads,
        old_label_bitwise_rebuild=read_json(OUT/'independent/labels-check.json')['original_label_rebuild_bitwise_sources'],
        train_only_rule_thresholds={k:v['threshold'] for k,v in read_json(OUT/'train-audit.json')['calibrations'].items()},
        direct_dense_rule_matches_original_confusions=True,independent_screen=decision,
        dense_training=dense_train,new_VGGT_extraction=read_json(OUT/'independent/VGGT-extraction.json'),
        note='No Bokeh prediction, no temporal claim; new-source check within same Sintel domain, not foundation-pretraining isolation')
    save_json(str(PREFIX)+'-verification.json',check)
    run.write_csv(str(PREFIX)+'-combined-summary.csv',summarize(all_rows))
    print('VERIFIED',arrays,'arrays,',reloads,'frame replays;',decision,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['download-new','prepare-new','extract-new','evaluate-original','evaluate-new','report','verify'])
    args=parser.parse_args()
    if args.phase=='download-new':
        from scripts.prepare_ours_geometry_pilot import download
        download(NEW)
    elif args.phase=='prepare-new':prepare_new()
    elif args.phase=='extract-new':run.extract(new_config(),ROOT/NEW['output'],'VGGT')
    elif args.phase=='evaluate-original':evaluate(False)
    elif args.phase=='evaluate-new':evaluate(True)
    elif args.phase=='report':report()
    else:verify()
