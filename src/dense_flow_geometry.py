"""Pose-constrained dense triangulation. No depth predictions or GT masks."""
import cv2
import numpy as np


def sample(image,xy):
    return cv2.remap(image,np.float32(xy[...,0]),np.float32(xy[...,1]),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def pair_depth(flow,back,K,first,second):
    """Closest-ray midpoint followed by the registered geometric checks."""
    h,w=flow.shape[:2];y,x=np.mgrid[:h,:w];uv=np.stack([x,y],-1).astype(np.float64);other=uv+flow
    inside=(other[...,0]>=0)&(other[...,0]<=w-1)&(other[...,1]>=0)&(other[...,1]<=h-1)
    fb=np.linalg.norm(flow+sample(back,other),axis=-1);valid=inside&(fb<=1.)
    left=np.concatenate([uv+.5,np.ones((h,w,1))],-1);right=np.concatenate([other+.5,np.ones((h,w,1))],-1)
    relative=np.linalg.inv(second)@first;t=relative[:3,3];skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    fundamental=np.linalg.inv(K).T@skew@relative[:3,:3]@np.linalg.inv(K)
    line=left@fundamental.T;reverse=right@fundamental;error=abs((right*line).sum(-1))
    epi=np.maximum(error/np.maximum(np.linalg.norm(line[...,:2],axis=-1),1e-12),error/np.maximum(np.linalg.norm(reverse[...,:2],axis=-1),1e-12));epi_valid=valid&(epi<=1.)
    rays1=(left@np.linalg.inv(K).T)@first[:3,:3].T;rays2=(right@np.linalg.inv(K).T)@second[:3,:3].T
    rays1/=np.linalg.norm(rays1,axis=-1,keepdims=True);rays2/=np.linalg.norm(rays2,axis=-1,keepdims=True)
    b=(rays1*rays2).sum(-1);displacement=second[:3,3]-first[:3,3];a=rays1@displacement;c=rays2@displacement
    s=(a-b*c)/np.maximum(1-b*b,1e-12);q=s*b-c
    world=.5*(first[:3,3]+s[...,None]*rays1+second[:3,3]+q[...,None]*rays2)
    ci=(world-first[:3,3])@first[:3,:3];cj=(world-second[:3,3])@second[:3,:3]
    angle=np.rad2deg(np.arccos(np.clip(b,-1,1)));valid=epi_valid&(ci[...,2]>0)&(cj[...,2]>0)&(angle>=.1)
    pi=ci@K.T;pj=cj@K.T;ri=np.linalg.norm(pi[...,:2]/np.maximum(pi[...,2:],1e-12)-left[...,:2],axis=-1);rj=np.linalg.norm(pj[...,:2]/np.maximum(pj[...,2:],1e-12)-right[...,:2],axis=-1)
    valid&=(np.maximum(ri,rj)<=1)&np.isfinite(ci[...,2])
    return np.where(valid,ci[...,2],np.nan),other,dict(pixels=h*w,in_bounds=int(inside.sum()),FB_pass=int((inside&(fb<=1)).sum()),epipolar_pass=int(epi_valid.sum()),pair_valid=int(valid.sum()))


def multiview_depth(pairs,K,target,poses):
    """Target-ray consensus from >=2 other views, with explicit reprojection."""
    depths=np.stack([p['depth'] for p in pairs]);count=np.isfinite(depths).sum(0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(depths,axis=0)
    consistent=np.isfinite(depths)&(abs(depths-median[None])<=np.maximum(.005,.02*median)[None]);h,w=median.shape
    y,x=np.mgrid[:h,:w];rays=np.stack([x+.5,y+.5,np.ones((h,w))],-1)@np.linalg.inv(K).T
    world=(rays*np.nan_to_num(median)[...,None])@target[:3,:3].T+target[:3,3]
    for n,p in enumerate(pairs):
        pose=poses[p['frame']];camera=(world-pose[:3,3])@pose[:3,:3];uvz=camera@K.T
        projected=uvz[...,:2]/np.maximum(uvz[...,2:],1e-12);error=np.linalg.norm(projected-(p['other_xy']+.5),axis=-1)
        consistent[n]&=(camera[...,2]>0)&(error<=1.)
    support=consistent.sum(0);valid=(support>=2)&np.isfinite(median)&(median>0)
    return np.where(valid,median,0).astype(np.float32),valid,support,count


def stratified_points(depth,valid,frame,variant):
    """Registered lattice, spatial heldout blocks and inverse-depth strata."""
    y,x=np.mgrid[:depth.shape[0],:depth.shape[1]];lattice=(x%4==2)&(y%4==2);valid=valid&lattice
    rng=np.random.default_rng(2026111032+frame+variant);blocks=rng.permutation(256);heldblocks=blocks[int(.8*len(blocks)):]
    held=np.isin((y//32)*16+x//32,heldblocks);rng=np.random.default_rng(2026111031+frame+variant);chosen=[]
    for subset in [depth<2.5,(depth>=2.5)&(depth<4),depth>=4]:
        ids=np.flatnonzero(valid&subset);ids=rng.permutation(ids)[:2000];chosen.append(ids)
    ids=np.concatenate(chosen);return np.stack([x.flat[ids],y.flat[ids]],-1).astype(np.float64),depth.flat[ids].astype(np.float64),held.flat[ids]
