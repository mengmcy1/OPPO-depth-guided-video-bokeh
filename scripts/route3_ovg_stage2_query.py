"""Frozen first-pass boundary Query and unchanged second-pass aperture Query."""
from functools import lru_cache

import cv2
import numpy as np
import torch

import route3_ovg_data as data
from route3_ovg_1b_common import SOURCE, read
from route3_ovg_footprint import query_net
from route3_ovg_refocus_query import original_views
from route3_s2a_data import query_input_512, query_memory, views_for as stored_views
from route3_s2a_learn import Store
from src.aperture_renderer import GaussianScene
from src.dual_layer import depth_band
from src.footprint_order import apply_footprint_order

Q_MIN = .5
QUERY = SOURCE/'V1/retrained/step-020000.pt'


def number(value, device='cuda'):
    return torch.as_tensor(value,device=device)


def present_rows(key):
    source=data.PREVIOUS/'geometry/original'/key
    path=data.PREVIOUS/'evaluation-input/original'/key/'512-13/manifest.json'
    return source,read(path) if path.exists() else None


def to_display(rgb):
    return np.where(rgb<=.0031308,12.92*rgb,1.055*rgb.clip(.0031308)**(1/2.4)-.055).clip(0,1)


def smaller_views(views,size):
    result=[]
    for v in views:
        n=v['depth'].shape[0]
        if n==size:
            result.append(v)
            continue
        row=dict(v,K=v['K'].copy())
        row['K'][:2]*=size/n
        for k in ('depth','confidence','edge','edge_distance','reprojection','FB','angle_radians',
                  'rgb_linear','rgb_srgb'):
            method=cv2.INTER_AREA if k.startswith('rgb') else cv2.INTER_NEAREST_EXACT
            field=v[k].astype(np.uint8) if k=='edge' else v[k]
            row[k]=cv2.resize(field,(size,size),interpolation=method)
            if k in ('edge_distance','reprojection','FB'):
                row[k]*=size/n
        row['edge']=row['edge'].astype(bool)
        result.append(row)
    return result


@lru_cache(maxsize=8)
def views_for(tier,key,size):
    if tier=='B':
        source,manifest=present_rows(key)
        frames=manifest['frames'] if manifest is not None else data.TRAIN_FRAMES
        if manifest is not None:
            observations=original_views(source,manifest)
        else:
            descriptor=dict(model=source/'models/SHARP',calibration=source/'models/calibration/SHARP',
                            confidence=source/'models/calibration')
            observations=stored_views(descriptor,frames,512)
        views=smaller_views(observations,size)
        # Verified memory IDs are encoded at 512². Low-res training still
        # uses the same observations, with coordinates recomputed at size.
        memory=query_memory(source,frames) if size==512 and (source/'memory/verification.npz').exists() else None
        return views,memory
    manifest=read(key)
    views=[]
    for idx in range(len(manifest['rows'])):
        src=data.gc_input(key,idx)
        depth=src['depth'][1]
        confidence=src['confidence'][1]
        inverse=np.divide(1,depth,out=np.zeros_like(depth),where=depth>0)
        edge=(cv2.dilate(inverse,np.ones((3,3),np.uint8))-
              cv2.erode(inverse,np.ones((3,3),np.uint8)))>.03*inverse
        from scipy.ndimage import distance_transform_edt
        views.append(dict(frame=idx,depth=np.where(confidence>0,depth,0),K=src['K'],
                          c2w=src['pose'],rgb_linear=src['rgb'],rgb_srgb=to_display(src['rgb']),
                          confidence=confidence,edge=edge,edge_distance=distance_transform_edt(~edge).astype(np.float32),
                          **{k:np.full(depth.shape,np.nan,np.float32) for k in ('reprojection','FB','angle_radians')}))
    return views,None


def context(views):
    return dict(images=number(np.stack([r['rgb_linear'] for r in views])).permute(0,3,1,2).contiguous(),
                c2w=number(np.stack([r['c2w'] for r in views]).astype(np.float32)),
                K=number(views[0]['K'].astype(np.float32)))


class FrozenQuery:
    def __init__(self,checkpoint=QUERY):
        self.network=query_net(checkpoint).requires_grad_(False)
        self.store=Store()

    @torch.no_grad()
    def _run(self, packed, views, frame, focus, aperture_scale):
        ctx=context(views)
        targets=packed['query_pixel']
        extent=packed['target_depth'].shape[1]
        conditions=np.c_[packed['query_xy']/extent,2/np.broadcast_to(focus,(len(targets),)),
                         np.full(len(targets),aperture_scale),
                         packed['target_depth'].ravel()[targets]/5,np.full(len(targets),frame/60)].astype(np.float32)
        chunks=[]
        for offset in range(0,len(targets),2048):
            sl=slice(offset,offset+2048)
            arrays=['features','candidate_valid','source_slots','source_xy','source_depth']
            args=[number(packed[k][sl]) for k in arrays]
            args[2]=args[2].long()
            args[3]=args[3].float()
            output=self.network(*args,number(conditions[sl]),ctx,number(packed['target_c2w'].astype(np.float32)))
            quality=(output['weights']*output['q_logits'].sigmoid()[:,None]).sum(-1)
            chunks.append({k:v.detach().cpu().numpy() for k,v in dict(
                xyz=output['means_camera'],depth=output['depth'],colors=output['colors'],
                opacity=output['opacities'],confidence=quality).items()})
        return {k:np.concatenate([r[k] for r in chunks]) for k in chunks[0]} if chunks else {}

    def first(self,tier,key,index,z1,z2):
        size=z1.shape[0]
        views,memory=views_for(tier,key,size)
        frame=data.TRAIN_FRAMES[index] if tier=='B' else index
        return self.first_from_views(views,memory,frame,z1,z2)

    def first_from_views(self,views,memory,frame,z1,z2):
        size=z1.shape[0]
        target=next(v for v in views if v['frame']==frame)
        mask=(z2>=1.03*z1)
        packed,_=query_input_512(views,dict(target,depth=z1),mask,8,size*size,memory,anchor=z2)
        pix=packed['query_pixel']
        arrays=dict(color=np.zeros((*z1.shape,3),np.float32),depth=z2.copy(),q=np.zeros_like(z1),
                    memory=np.zeros((*z1.shape,3),np.float32),memory_q=np.zeros_like(z1))
        if not len(pix):
            return arrays
        # Candidate index zero is the closest observation to the depth anchor.
        # M0 comes from original observed color, separate from the learned Query color.
        arrays['memory'].reshape(-1,3)[pix]=packed['features'][:,0,4:7]
        arrays['memory_q'].ravel()[pix]=packed['features'][:,0,11]
        raw=self._run(packed,views,frame,z2.ravel()[pix],0.)
        valid=(raw['opacity']>1e-4)&(raw['depth']>=1.03*z1.ravel()[pix,None])
        with_quality=np.where(valid,raw['confidence'],-1.)
        heads=with_quality.argmax(-1)
        i=np.arange(len(pix))
        keep=valid[i,heads]
        i,heads,pix=i[keep],heads[keep],pix[keep]
        arrays['color'].reshape(-1,3)[pix]=raw['colors'][i,heads]
        arrays['depth'].ravel()[pix]=raw['depth'][i,heads]
        arrays['q'].ravel()[pix]=raw['confidence'][i,heads]
        return arrays

    def second(self,tier,key,index,z,focus,fstop):
        size=z.shape[0]
        views,memory=views_for(tier,key,size)
        frame=data.TRAIN_FRAMES[index] if tier=='B' else index
        return self.second_from_views(views,memory,frame,z,focus,fstop)

    def second_from_views(self,views,memory,frame,z,focus,fstop):
        size=z.shape[0]
        target=next(v for v in views if v['frame']==frame)
        K=target['K']
        band=depth_band(z,K[0,0],focus,.05/(2*fstop))[0]
        packed,_=query_input_512(views,dict(target,depth=z),band,8,size*size,memory)
        output=self._run(packed,views,frame,focus,50*.05/(2*fstop))
        if not output:
            return None
        xyz=output['xyz'].reshape(-1,3)
        rgb=output['colors'].reshape(-1,3)
        opacity=output['opacity'].ravel()
        uvz=xyz@K.T
        uv=uvz[:,:2]/np.maximum(uvz[:,2:],1e-6)
        valid=(xyz[:,2]>0)&(opacity>1e-4)&(uv>=0).all(-1)&(uv<size).all(-1)
        pix=uv[valid].astype(int)
        valid[np.flatnonzero(valid)]=band[pix[:,1],pix[:,0]]
        if not valid.any():
            return None
        xyz,rgb,opacity=map(number,(xyz[valid],rgb[valid],opacity[valid]))
        sigma=xyz[:,2]/K[0,0]*.7
        scale=torch.stack([sigma,sigma,sigma*.01],-1)
        quat=xyz.new_zeros((len(xyz),4))
        quat[:,0]=1
        cloud=GaussianScene(xyz,quat,scale,opacity,rgb,torch.ones_like(opacity,dtype=torch.long))
        return apply_footprint_order(cloud,number(z),number(K))[0]
