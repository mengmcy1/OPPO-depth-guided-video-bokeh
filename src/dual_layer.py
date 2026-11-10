"""Untrained visible-pixel warp and depth/CoC-only aperture bands.

No hidden geometry, masks, texture or future frame information enters band
construction or the visible-pixel warp. Cameras use OpenCV pixel centers.
"""
import math
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from .aperture_renderer import aperture_disk,linear_to_srgb,srgb_to_linear


def depth_band(depth,fx,focus,aperture,multiplier=1.5,relative_jump=.03):
    """Union variable CoC footprints around visible inverse-depth jumps.

    A3% relative depth step is fixed, not fitted to hidden GT. Unknown sky is
    the common far-field approximation, never a valid recovered surface.
    """
    import cv2
    z=np.asarray(depth,np.float32);inv=1/np.maximum(z,1e-6)
    local_max=cv2.dilate(inv,np.ones((3,3),np.uint8))
    local_min=cv2.erode(inv,np.ones((3,3),np.uint8))
    edge=(local_max-local_min)>relative_jump*np.maximum(local_max,1e-6)
    radius=abs(float(fx)*float(aperture)*(inv-1/float(focus)))
    footprint=np.ceil(multiplier*np.maximum(cv2.dilate(radius,np.ones((3,3),np.uint8)),1.)).astype(np.int32)
    band=np.zeros(z.shape,np.uint8)
    for r in np.unique(footprint[edge]):
        seed=(edge&(footprint==r)).astype(np.uint8)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(r)*2+1,int(r)*2+1))
        band|=cv2.dilate(seed,kernel)
    return band.astype(bool),edge,radius


def warp_visible(rgb_linear:Tensor,depth:Tensor,K:Tensor,focus,xy:Tensor,
                 soft_z_relative=.001,depth_slope_correction=False):
    """Bilinear forward splat with per-destination soft z-buffer.

    Colors are unpremultiplied C_v; alpha is clipped geometric coverage. No
    forced opacity in warp holes. A local minimum-depth reduction stabilizes
    the soft z weights, without changing displacement or the exposure.
    """
    h,w=depth.shape;device,dtype=depth.device,depth.dtype
    yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=dtype),torch.arange(w,device=device,dtype=dtype),indexing='ij')
    factor=torch.stack((K[0,0],K[1,1]))*xy
    # Forward image motion has opposite sign to residual-fill lookup.
    uv=torch.stack((xx,yy),-1)+factor*(1/torch.as_tensor(focus,device=device,dtype=dtype)-1/depth)[...,None]
    low=uv.floor();frac=uv-low
    records=[];z=depth.reshape(-1);color=rgb_linear.reshape(-1,3)
    gradient=None
    if depth_slope_correction:
        # Source pixel-center depths differ even on one slanted plane. Compare
        # their extrapolated destination-ray depths, not raw source depths.
        inv=torch.where(depth>0,1/depth,torch.zeros_like(depth));derivatives=[]
        for dim,coord,extent in [(1,xx,w),(0,yy,h)]:
            left=inv.roll(1,dim);right=inv.roll(-1,dim)
            a=(left>0)&(depth>0)&(coord>0)&((left-inv).abs()<=.03*inv.abs())
            b=(right>0)&(depth>0)&(coord<extent-1)&((right-inv).abs()<=.03*inv.abs())
            derivative=torch.where(a&b,(right-left)/2,torch.where(b,right-inv,torch.where(a,inv-left,torch.zeros_like(inv))))
            derivatives.append(derivative)
        g=torch.stack(derivatives,-1);denominator=(1-(g*factor).sum(-1)).clamp_min(.2)
        gradient=(g/denominator[...,None]).reshape(-1,2)
    zmin=torch.full((h*w,),float('inf'),device=device,dtype=dtype)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        u=low[...,0].long()+dx;v=low[...,1].long()+dy
        weight=(frac[...,0] if dx else 1-frac[...,0])*(frac[...,1] if dy else 1-frac[...,1])
        valid=(u>=0)&(u<w)&(v>=0)&(v<h)&(weight>0)&torch.isfinite(depth)&(depth>0)
        source=torch.nonzero(valid.reshape(-1),as_tuple=False).flatten();destination=(v[valid]*w+u[valid]).long()
        weight=weight[valid]
        buffer_depth=z[source]
        if gradient is not None:
            displacement=torch.stack([u[valid],v[valid]],-1).to(dtype)-uv.reshape(-1,2)[source]
            inv_at_destination=1/z[source]+(gradient[source]*displacement).sum(-1)
            buffer_depth=torch.where(inv_at_destination>1e-8,1/inv_at_destination,z[source])
        zmin.scatter_reduce_(0,destination,buffer_depth,reduce='amin',include_self=True)
        records.append((source,destination,weight,buffer_depth))
    mass=torch.zeros(h*w,device=device,dtype=dtype);rgb=torch.zeros(h*w,3,device=device,dtype=dtype);zmass=torch.zeros_like(mass)
    for src,dst,weight,buffer_depth in records:
        visibility=torch.exp(-(buffer_depth-zmin[dst]).clamp_min(0)/(soft_z_relative*zmin[dst].clamp_min(1e-3)))
        weight=weight*visibility
        mass.scatter_add_(0,dst,weight);rgb.index_add_(0,dst,weight[:,None]*color[src]);zmass.scatter_add_(0,dst,weight*buffer_depth)
    alpha=mass.clamp(0,1).reshape(h,w)
    return dict(color=(rgb/mass[:,None].clamp_min(1e-12)).reshape(h,w,3),
        depth=(zmass/mass.clamp_min(1e-12)).reshape(h,w),alpha=alpha)


def compose_visible_hidden(visible,hidden_premult,hidden_depth,hidden_transmission,fill,
                           depth_tolerance=1e-4):
    """Opaque visible first surface, then depth-ordered hidden GS and fill.

    The stipulated alpha_v formula is valid only when hidden depth is behind
    visible depth. Unexpected in-front hidden samples are counted; they use
    the reverse order instead of silently being obscured by the visible map.
    """
    av=visible['alpha'];cv=visible['color'];zv=visible['depth'];th=hidden_transmission
    ah=1-th
    behind=hidden_premult+th[...,None]*fill
    visible_first=av[...,None]*cv+(1-av[...,None])*behind
    hidden_first=hidden_premult+th[...,None]*(av[...,None]*cv+(1-av[...,None])*fill)
    reverse=(av>1e-6)&(ah>1e-6)&(hidden_depth>0)&(hidden_depth<zv-depth_tolerance)
    return torch.where(reverse[...,None],hidden_first,visible_first),reverse


def render_pixels(rgb_linear,depth,K,focus,aperture,samples,fill_samples,
                  hidden_samples=None,soft_z_relative=.001,depth_slope_correction=False):
    disk=aperture_disk(samples,aperture,device=depth.device,dtype=depth.dtype)
    image=torch.zeros_like(rgb_linear);alpha=torch.zeros_like(depth);reverse_count=0
    for i,xy in enumerate(disk):
        visible=warp_visible(rgb_linear,depth,K,focus,xy,soft_z_relative,depth_slope_correction)
        if hidden_samples is None:
            current=visible['alpha'][...,None]*visible['color']+(1-visible['alpha'][...,None])*fill_samples[i]
        else:
            hid=hidden_samples[i]
            current,reverse=compose_visible_hidden(visible,hid['rgb_linear'],hid['depth'],1-hid['alpha'],fill_samples[i])
            reverse_count+=int(reverse.sum())
        image+=current;alpha+=visible['alpha']
    return dict(rgb_linear=image/samples,alpha=alpha/samples,reverse_order_ray_pixels=reverse_count,aperture_xy=disk)
