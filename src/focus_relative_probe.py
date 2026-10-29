"""Architecture V2 Stage0: independent axial-depth labels and frozen-feature probe.

This version namespace is unrelated to the old disparity Student or subject clamp.
No bokeh generation, fitted d_ref, GT focus as an inference input or dev tuning.
"""
import hashlib
import json
from pathlib import Path
import struct

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f'Preserve existing artifact: {path}'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def depth_read(path):
    raw = Path(path).read_bytes()
    assert raw[:4] == b'PIEH'
    w,h = struct.unpack('<ii',raw[4:12])
    assert len(raw) == 12+w*h*4
    return np.frombuffer(raw, dtype='<f4',offset=12).reshape(h,w).copy()


def coordinate_indices(native_wh, model_wh):
    nw,nh = native_wh
    w,h = model_wh
    x = np.floor((np.arange(w)+.5)*nw/w).astype(int)
    y = np.floor((np.arange(h)+.5)*nh/h).astype(int)
    assert x.min()>=0 and x.max()<nw and y.min()>=0 and y.max()<nh
    return x,y


def focus_boxes(config):
    w,h = config['preprocessing']['model_wh']
    fw,fh = config['focus']['size_display_wh']
    boxes=[]
    for x,y in config['focus']['centers_normalized_xy']:
        x0,y0 = int(round(x*w-fw/2)),int(round(y*h-fh/2))
        assert x0>=0 and y0>=0 and x0+fw<=w and y0+fh<=h
        boxes.append([x0,y0,fw,fh])
    return boxes


def make_focus_masks(config):
    w,h = config['preprocessing']['model_wh']
    masks = np.zeros((len(focus_boxes(config)),h,w),np.float32)
    for i,(x,y,bw,bh) in enumerate(focus_boxes(config)):
        masks[i,y:y+bh,x:x+bw]=1
    return masks


def build_labels(root, config, config_path=None):
    out = root/config['output']
    dest=out/'data'
    dest.mkdir(parents=True,exist_ok=True)
    xindex,yindex=coordinate_indices(config['preprocessing']['native_wh'],config['preprocessing']['model_wh'])
    w,h = config['preprocessing']['model_wh']
    nw,nh = config['preprocessing']['native_wh']
    eps = config['labels']['epsilon']
    manifests=[]; counts=[]
    for scene in config['sequences']:
        name=scene['sequence'];path=dest/(name+'.npz')
        if path.exists():
            manifests.extend(read_json(dest/(name+'-manifest.json')))
            continue
        yy=[];valids=[];boundaries=[];zs=[];vds=[];scene_rows=[]
        with np.load(root/config['vda_root']/name/'prediction.npz') as f:
            vda = f['disparity_normalized']
        for t,frame in enumerate(scene['frames']):
            source=root/config['raw_root']/'training'
            z=depth_read(source/'depth'/name/f'frame_{frame:04d}.dpt')
            assert z.shape==(nh,nw)
            valid=np.isfinite(z)&(z>0)
            small=z[yindex[:,None],xindex[None,:]]
            vm=valid[yindex[:,None],xindex[None,:]]
            logz=np.log(np.where(vm,small,1)).astype(np.float32)
            complete=cv2.erode(vm.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            edge=complete & ((cv2.dilate(logz,np.ones((3,3),np.uint8))-cv2.erode(logz,np.ones((3,3),np.uint8)))>config['boundary']['threshold'])
            edge=(cv2.dilate(edge.astype(np.uint8),np.ones((3,3),np.uint8))>0)&vm
            labels=[]
            for fi,(fx,fy,fw,fh) in enumerate(focus_boxes(config)):
                a,b=int(np.floor(fx*nw/w)),int(np.floor(fy*nh/h))
                c,d=int(np.ceil((fx+fw)*nw/w)),int(np.ceil((fy+fh)*nh/h))
                focus_native=valid[b:d,a:c];focus_display=vm[fy:fy+fh,fx:fx+fw]
                ok=focus_display.sum()>=config['focus']['minimum_valid_pixels'] and focus_display.mean()>=config['focus']['minimum_valid_fraction'] and focus_native.any()
                ref=float(np.median(z[b:d,a:c][focus_native])) if ok else None
                label=np.full((h,w),255,np.uint8)
                if ok:
                    r=np.log(ref/np.where(vm,small,1))
                    label[vm]=1;label[vm & (r>eps)]=0;label[vm & (r < -eps)]=2
                labels.append(label)
                row=dict(sequence=name,source_family=scene['source_family'],split=scene['split'],frame_id=frame,array_index=t,focus_id=fi,
                         RGB=str((source/'clean'/name/f'frame_{frame:04d}.png').relative_to(root)),
                         axial_Z=str((source/'depth'/name/f'frame_{frame:04d}.dpt').relative_to(root)),
                         display_xywh=[fx,fy,fw,fh],native_xyxy=[a,b,c,d],focus_valid_pixels_display=int(focus_display.sum()),
                         focus_valid_pixels_native=int(focus_native.sum()),GT_Z_focus=ref,valid_problem=bool(ok),skip_reason='' if ok else 'focus numeric validity below frozen requirements',
                         label_counts=[int((label==i).sum()) for i in range(3)],valid_pixels=int(vm.sum()),boundary_pixels=int(edge.sum()))
                scene_rows.append(row)
            yy.append(np.stack(labels));valids.append(vm);boundaries.append(edge);zs.append(small)
            vds.append(vda[t][yindex[:,None],xindex[None,:]])
        np.savez_compressed(path,labels=np.stack(yy),valid=np.stack(valids),boundary=np.stack(boundaries),Z=np.stack(zs),vda=np.stack(vds))
        save_json(dest/(name+'-manifest.json'),scene_rows);manifests.extend(scene_rows)
    train=[r for r in manifests if r['split']=='pilot_train' and r['valid_problem']]
    class_counts=np.sum([r['label_counts'] for r in train],axis=0)
    assert (class_counts>0).all()
    weights=(class_counts/class_counts.sum())**-.5;weights/=weights.mean()
    plan=dict(config_sha256=sha(config_path or root/'configs/v2-stage0-20261029.json'),questions=manifests,
              frames=150,sources=6,train_questions=len(train),dev_questions=sum(r['split']=='pilot_dev' and r['valid_problem'] for r in manifests),
              training_class_counts=class_counts.tolist(),training_loss_weights=weights.tolist(),
              coordinate_check=dict(native_wh=[nw,nh],model_wh=[w,h],xindex_first_last=xindex[[0,-1]].tolist(),yindex_first_last=yindex[[0,-1]].tolist(),
                                    no_depth_averaging=True,no_crop_no_padding=True,no_GT_model_inputs=True),
              validity_limit='finite positive raw axial Z only; large/far depths retained; no independent transparent/mixed-pixel exclusion available')
    save_json(root/(config['record_prefix']+'-manifest.json'),plan)
    return plan


def confusion(y,p,mask=None):
    m=(y<3) if mask is None else (y<3)&mask
    return np.bincount(3*y[m].astype(np.int64)+p[m].astype(np.int64),minlength=9).reshape(3,3)


def metrics(cm):
    cm=np.asarray(cm,dtype=np.float64)
    diag=np.diag(cm);support=cm.sum(1);pred=cm.sum(0)
    precision=np.divide(diag,pred,out=np.zeros(3),where=pred>0)
    recall=np.divide(diag,support,out=np.zeros(3),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(3),where=precision+recall>0)
    return dict(Macro_F1=float(f1.mean()),Balanced_Accuracy=float(recall.mean()),
                precision=precision.tolist(),recall=recall.tolist(),F1=f1.tolist(),class_support=support.astype(np.int64).tolist(),
                valid_pixels=int(cm.sum()),confusion=cm.astype(np.int64).tolist(),absent_class_zero_division_policy='fixed3 denominator, absent GT class recall/F1=0')


def rule_score(values,box,is_depth):
    x,y,w,h=box
    reference=float(np.median(values[y:y+h,x:x+w]))
    if is_depth:
        assert np.isfinite(values).all() and (values>0).all() and reference>0
        return np.log(reference/values)
    return values-reference


def threshold_calibration(root,c,method):
    ts=np.asarray(c['models'][method]['threshold_candidates'])
    scores=[];per_source={}
    for scene in c['sequences']:
        if scene['split']!='pilot_train':continue
        name=scene['sequence'];data=np.load(root/c['output']/'data'/f'{name}.npz')
        depth=np.load(root/c['output']/'features'/name/'depth.npy',mmap_mode='r') if method=='P2' else data['vda']
        hist=np.zeros((3,2,len(ts)+1),np.int64);total=np.zeros(3,np.int64)
        for t in range(25):
            for f,box in enumerate(focus_boxes(c)):
                y=data['labels'][t,f];m=y<3
                score=rule_score(depth[t],box,method=='P2')[m];gt=y[m]
                total+=np.bincount(gt,minlength=3)
                for sign,sm in enumerate([score>0,score<0]):
                    bucket=np.searchsorted(ts,np.abs(score[sm]),side='left')
                    hist[:,sign,:]+=np.bincount(gt[sm]*(len(ts)+1)+bucket,minlength=3*(len(ts)+1)).reshape(3,-1)
        tail=np.cumsum(hist[:,:,::-1],axis=2)[:,:,::-1]
        cms=[]
        for i in range(len(ts)):
            cm=np.zeros((3,3),np.int64);cm[:,0]=tail[:,0,i+1];cm[:,2]=tail[:,1,i+1];cm[:,1]=total-cm[:,0]-cm[:,2];cms.append(cm)
        per_source[name]=cms
    for i,threshold in enumerate(ts):
        scores.append(dict(threshold=float(threshold),source_equal_Macro_F1=float(np.mean([metrics(v[i])['Macro_F1'] for v in per_source.values()]))))
    best=max(range(len(scores)),key=lambda i:(scores[i]['source_equal_Macro_F1'],-scores[i]['threshold']))
    return dict(method=method,threshold=scores[best]['threshold'],selection='four training sources only; fixed candidate grid; tie smallest',training_curve=scores,
                per_source={k:metrics(v[best]) for k,v in per_source.items()})


class RelationProbe(nn.Module):
    def __init__(self,channels,config):
        super().__init__();pc=config['head']['projection_channels'];h1,h2=config['head']['relation_hidden']
        self.channels=channels
        self.projection=nn.Conv2d(channels,pc,1)
        self.relation=nn.Sequential(nn.Conv2d(pc*4,h1,1),nn.GELU(),nn.Conv2d(h1,h2,1),nn.GELU(),nn.Conv2d(h2,3,1))

    def forward(self,features,focus):
        if self.channels>1:
            features=F.normalize(features.float(),dim=1)*self.channels**.5
        else:
            features=features.float()
        f=self.projection(features)
        mask=F.interpolate(focus.float(),size=f.shape[-2:],mode='area')
        q=(f*mask).sum((-2,-1),keepdim=True)/mask.sum((-2,-1),keepdim=True).clamp_min(1e-8)
        q=q.expand_as(f)
        logits=self.relation(torch.cat([f,q,f-q,f*q],dim=1))
        return F.interpolate(logits,size=focus.shape[-2:],mode='bilinear',align_corners=False)


def self_checks():
    y=np.array([[0,1,2,255]],np.uint8);p=np.array([[0,1,2,0]],np.uint8)
    assert metrics(confusion(y,p))['Macro_F1']==1
    assert metrics([[1,0,0],[0,0,0],[0,0,0]])['Macro_F1']==1/3
    v=np.array([[1.,2.,4.]],np.float32)
    score=rule_score(v,[1,0,1,1],True)
    assert score[0,0]>0 and score[0,1]==0 and score[0,2]<0
    # Label geometry never uses old fitted disparity reference.
    x,yi=coordinate_indices([1024,436],[518,224]);assert len(x)==518 and len(yi)==224
    return dict(confusion_and_absent_class=True,depth_direction=True,coordinate_mapping=True)
