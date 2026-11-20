"""Stage 1a samples with a physically separate input and supervision mapping."""
import functools
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from route3_ovg_subpixel import CONFIG, OUT, PREVIOUS, ROOT, TRAIN_FRAMES
from route3_ovg_tartan import depth_from_png

sys.path.insert(0,str(ROOT))
PRIORS=('SHARP','DepthSplat','triangulation')
LABELS=('depth1','depth2','alpha','normal','valid','valid_alpha','valid_dual','valid_normal','depth_boundary')


def read(path):
    return json.loads(path.read_text())


def load(path, keys=None):
    with np.load(path) as d:
        return {k:d[k] for k in (keys or d.files)}


def srgb_to_linear(v):
    return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4).astype(np.float32)


def confidence(value, valid):
    return np.where(valid, np.clip(value,0,1),0).astype(np.float32)


def b_folder(name):
    return PREVIOUS/'geometry'/'original'/name


@functools.lru_cache(maxsize=8)
def b_foreground_reference(name, frame):
    """Read GT identity from labels only, outside input construction."""
    expanded = ROOT/'data/processed/route3-v2data-2026-11-20/B/units/development'/name/'labels'
    current = expanded/f'f{frame:03d}-pinhole.npz'
    if current.is_file():
        root, center = expanded, current
    else:
        truth = PREVIOUS/'truth'/name
        root, center = truth, truth/'geometry'/f'f{frame:03d}-center.npz'
    row = load(center, ('depth_axial_m','object_id','center_visible'))
    return dict(gt=row['depth_axial_m'],object_id=row['object_id'],valid=row['center_visible'],
                foreground=load(root/'scene-geometry.npz',('foreground_object_ids',))['foreground_object_ids'])


def b_edge_reference(item, target):
    _, name, index = item
    frame = TRAIN_FRAMES[index]
    row = b_foreground_reference(name,frame)
    return dict(row, valid=row['valid'] & target['valid'][0].cpu().numpy(),
                boundary=target['depth_boundary'][0].cpu().numpy())


@functools.lru_cache(maxsize=48)
def b_input(name, frame):
    root=b_folder(name)
    f=f'f{frame:03d}'
    calibrated=root/'models/calibration'
    sharp=load(calibrated/'SHARP'/(f+'.npz'))
    ds=load(PREVIOUS/'geometry'/'depthsplat-official'/name/'depthsplat'/(f+'.npz'))
    ref=load(calibrated/(f+'-reference.npz'))
    q=load(calibrated/(f+'-confidence.npz'))
    rgb=load(root/'models/SHARP'/(f+'.npz'),('rgb_linear',))['rgb_linear']
    depth=np.stack([sharp['depth'],ds['depth'],ref['depth']]).astype(np.float32)
    valid=np.isfinite(depth)&(depth>=.05)&(depth<=200)
    a=confidence(np.maximum(q['confidence'],.2),valid[0]&sharp['dominant_valid'])
    b=confidence(ds['support']/7,valid[1]&ds['consistent'])
    c=confidence(q['confidence'],valid[2]&ref['valid']&(ref['support']>=2)&q['supported'])
    return dict(rgb=rgb,K=sharp['K'].astype(np.float32),pose=sharp['c2w'].astype(np.float32),
                depth=np.where(valid,depth,0),confidence=np.stack([a,b,c]))


def normal_from_depth(z,K,valid):
    h,w=z.shape
    y,x=np.mgrid[:h,:w]
    surface=(np.dstack([x+.5,y+.5,np.ones_like(x)])@np.linalg.inv(K).T)*z[...,None]
    horizontal=surface[1:-1,2:]-surface[1:-1,:-2]
    vertical=surface[2:,1:-1]-surface[:-2,1:-1]
    n=-np.cross(horizontal,vertical)
    length=np.linalg.norm(n,axis=-1)
    mid=z[1:-1,1:-1]
    pass_neighborhood=valid[1:-1,1:-1].copy()
    for other in [z[1:-1,2:],z[1:-1,:-2],z[2:,1:-1],z[:-2,1:-1]]:
        pass_neighborhood &= np.isfinite(other)&(abs(mid-other)<.03*mid)
    mask=np.zeros_like(valid)
    mask[1:-1,1:-1]=pass_neighborhood&(length>1e-9)
    normal=np.zeros((*z.shape,3),np.float32)
    normal[1:-1,1:-1]=np.nan_to_num(n/np.maximum(length[...,None],1e-9))
    return normal,mask


@functools.lru_cache(maxsize=64)
def gc_input(window_path, index):
    path=Path(window_path)
    meta=read(path)
    row=meta['rows'][index]
    image=cv2.imread(str(path.parent/'input-resized'/f'f{index:03d}.png'))
    if image is None:
        raise FileNotFoundError(path.parent/'input-resized'/f'f{index:03d}.png')
    size=CONFIG['size']
    raw=cv2.imread(str(ROOT/row['image']))
    if raw is None:
        raise FileNotFoundError(ROOT/row['image'])
    K=np.array(row['K'],np.float32)
    K[:2]*=np.array([size/raw.shape[1],size/raw.shape[0]])[:,None]
    pre=load(path.parent/'prior'/f'f{index:03d}.npz')
    depth=pre['depth'].astype(np.float32)
    d=np.zeros((3,*depth.shape),np.float32)
    c=np.zeros_like(d)
    d[1]=depth
    c[1]=confidence(pre['support']/max(1,min(7,len(meta['rows'])-1)),pre['consistent']&np.isfinite(depth)&(depth>0))
    return dict(rgb=srgb_to_linear(image[...,::-1].astype(np.float32)/255),
                K=K,pose=np.array(row['c2w'],np.float32),depth=d,confidence=c)


def no_label(z, K):
    valid=np.isfinite(z)&(z>=.05)&(z<=200)
    z=np.where(valid,z,0).astype(np.float32)
    normal,n_valid=normal_from_depth(z,K,valid)
    missing=np.zeros(z.shape,bool)
    inverse=np.divide(1,z,out=np.zeros_like(z),where=valid)
    edge=(cv2.dilate(inverse,np.ones((3,3),np.uint8))-cv2.erode(inverse,np.ones((3,3),np.uint8)))>.03*inverse
    return dict(depth1=z,depth2=np.zeros_like(z),alpha=np.zeros_like(z),normal=normal,
                valid=valid,valid_alpha=missing,valid_dual=missing,valid_normal=n_valid,depth_boundary=edge)


def input_record(current, neighbors):
    def single_prior(row):
        q=row['confidence']
        inv=np.divide(1,row['depth'],out=np.zeros_like(q),where=q>0)
        total=q.sum(0)
        reciprocal=(q*inv).sum(0)/np.maximum(total,1e-8)
        depth=np.divide(1,reciprocal,out=np.zeros_like(reciprocal),where=reciprocal>0)
        return depth,q.max(0)
    evidence=[single_prior(v) for v in neighbors]
    return dict(rgb=current['rgb'].transpose(2,0,1),prior_depth=current['depth'],
                prior_conf=current['confidence'],K=current['K'],pose=current['pose'],
                motion=np.zeros(current['rgb'].shape[:2],bool),
                # The first B/C and selected TartanAir environments are
                # static captures by construction. No GT segmentation read.
                neighbors=np.stack([v['rgb'].transpose(2,0,1) for v in neighbors]),
                neighbor_K=np.stack([v['K'] for v in neighbors]),
                neighbor_pose=np.stack([v['pose'] for v in neighbors]),
                neighbor_depth=np.stack([v[0] for v in evidence]),
                neighbor_conf=np.stack([v[1] for v in evidence]))


def neighbors(index, n):
    return sorted((j for j in range(n) if index!=j),key=lambda j:(abs(index-j),j))[:2]


def sample(item, device, *, b_loader=b_input):
    tier,key,index=item
    if tier=='B':
        frame=TRAIN_FRAMES[index]
        # Construct the observed-input dictionary before reading supervision.
        current=b_loader(key,frame)
        other=[b_loader(key,TRAIN_FRAMES[j]) for j in neighbors(index,len(TRAIN_FRAMES))]
        inputs=input_record(current,other)
        target=load(OUT/'B'/key/'pinhole-subpixels'/f'f{frame:03d}.npz',LABELS)
    else:
        filename=Path(key)
        m=read(filename); rows=m['rows']
        current=gc_input(key,index)
        other=[gc_input(key,j) for j in neighbors(index,len(rows))]
        inputs=input_record(current,other)
        row=rows[index]
        if tier=='C':
            target=load(ROOT/row['label'],LABELS)
        elif tier=='G':
            raw=depth_from_png(ROOT/row['depth'])
            z=cv2.resize(raw,current['rgb'].shape[:2][::-1],interpolation=cv2.INTER_NEAREST_EXACT)
            target=no_label(z,current['K'])
        else:
            raise ValueError(tier)
    target['normal']=target['normal'].transpose(2,0,1)
    def tensor_map(d):
        return {k:torch.as_tensor(v[None].copy(),device=device) for k,v in d.items()}
    return tensor_map(inputs),tensor_map(target)


def inventory(role):
    split=read(PREVIOUS/'split.json')
    names=split['train' if role=='train' else 'heldout']
    data={'B':[('B',s,j) for s in names for j,f in enumerate(TRAIN_FRAMES)
               if (OUT/'B'/s/'pinhole-subpixels'/f'f{f:03d}.npz').exists()]}
    if role!='train':
        return data
    sealed=read(ROOT/'03_实验记录/records/route3-ovg-1a-2026-11-18-independent-test.json')['forbidden_train']
    if set(names)&set(sealed['scene_ids']):
        raise ValueError('B split overlaps sealed test data')
    for k in ('G','C'):
        data[k]=[]
        for p in sorted((OUT/k/'windows').glob('*/manifest.json')):
            meta=read(p)
            if meta['scene'] in sealed['scene_ids'] or meta.get('seed') in sealed['seeds']:
                raise ValueError(f'Train dataset overlaps sealed item {meta["scene"]}')
            if set(x['asset'] for x in meta.get('assets',[]))&set(sealed['asset_ids']):
                raise ValueError(f'Train source asset overlaps sealed item {meta["scene"]}')
            if k=='C' and not (OUT/'C/mesh-check.json').exists():
                continue
            rows=[(k,str(p),i) for i in range(len(meta['rows']))
                  if (p.parent/'prior'/f'f{i:03d}.npz').exists()]
            if len(rows)==len(meta['rows']):
                data[k].extend(rows)
    return data
