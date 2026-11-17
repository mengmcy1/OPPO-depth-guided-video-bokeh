"""Appendix13 fixed center-pixel ordering and exact per-ray fallback.

Ordering moves only offending centers along their original camera rays.
Colors, covariance scales, rotations and opacity are preserved, as specified.
"""
import numpy as np
import torch
from .aperture_renderer import GaussianScene,aperture_disk
from .layered_aperture import aperture_fill_samples


def order_hidden(scene,visible_depth,K,rule,background_depth=None):
    if scene is None:
        return None,dict(total=0,violating=0,moved=0,dropped=0)
    if rule not in ['none','O1','O2']:
        raise ValueError(rule)
    h,w=visible_depth.shape
    z=scene.means[:,2]
    projected=scene.means@K.T
    uv=projected[:,:2]/projected[:,2:].clamp_min(1e-8)
    inside=(z>0)&torch.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    x=uv[:,0].floor().long().clamp(0,w-1)
    y=uv[:,1].floor().long().clamp(0,h-1)
    visible=visible_depth[y,x]
    valid=inside&torch.isfinite(visible)&(visible>0)
    violating=valid&(z<visible*1.03)
    target=z.clone()
    keep=torch.ones_like(z,dtype=torch.bool)
    if rule=='O1':
        target=torch.where(violating,visible*1.03,target)
    elif rule=='O2':
        if background_depth is None:
            raise ValueError('O2 requires input-only 24px maximum-filter depth')
        background=background_depth[y,x]
        accept=torch.isfinite(background)&(background>=visible*1.03)
        keep=~violating|accept
        target=torch.where(violating&accept,background,target)
    moved=keep&violating&(rule!='none')
    means=scene.means.clone()
    means[moved]*=(target[moved]/z[moved])[:,None]
    fields={k:getattr(scene,k)[keep] for k in scene.__dataclass_fields__}
    fields['means']=means[keep]
    audit=dict(rule=rule,total=len(z),valid_center_pixels=int(valid.sum()),outside_or_no_reference=int((~valid).sum()),violating=int(violating.sum()),moved=int(moved.sum()),dropped=int((~keep).sum()),scale_rotation_opacity_color_unchanged=True,center_motion='along camera ray;scales remain fixed',background_filter_radius_px=24 if rule=='O2' else None)
    return GaussianScene(**fields),audit


def ray_samples(scene,K,focus,aperture,count,fill,background_depth):
    """A's final per-ray RGB, including its existing LaMa residual background."""
    from gsplat import rasterization
    h,w=background_depth.shape
    fills,_,_=aperture_fill_samples(fill,background_depth,K,focus,aperture,count)
    result=[]
    for i,xy in enumerate(aperture_disk(count,aperture,device=K.device,dtype=K.dtype)):
        view=torch.eye(4,device=K.device)
        view[:2,3]-=xy
        intr=K.clone()
        intr[:2,2]+=K.diag()[:2]*xy/focus
        image,a,_=rasterization(scene.means,scene.quats,scene.scales,scene.opacities,scene.colors,view[None],intr[None],w,h,packed=False,render_mode='RGB',rasterize_mode='antialiased',near_plane=1e-4,far_plane=1e5,eps2d=.0015)
        result.append(image[0]+(1-a[0])*fills[i])
    return torch.stack(result)


def render_fallbacks(scene,K,focus,aperture,count,fill,background_depth,A_samples):
    """Same native surface raster, independently composited with F1 or F2.

    F2 is sample_backgrounds: it never uses an averaged A image as background.
    This inference helper matches the renderer's residual-alpha convention.
    """
    from gsplat import rasterization
    h,w=background_depth.shape
    fills,_,_=aperture_fill_samples(fill,background_depth,K,focus,aperture,count)
    assert A_samples.shape==fills.shape
    rgb1=torch.zeros_like(fill)
    rgb2=torch.zeros_like(fill)
    alpha=torch.zeros_like(background_depth)
    for i,xy in enumerate(aperture_disk(count,aperture,device=K.device,dtype=K.dtype)):
        view=torch.eye(4,device=K.device)
        view[:2,3]-=xy
        intr=K.clone()
        intr[:2,2]+=K.diag()[:2]*xy/focus
        image,a,_=rasterization(scene.means,scene.quats,scene.scales,scene.opacities,scene.colors,view[None],intr[None],w,h,packed=False,render_mode='RGB',rasterize_mode='antialiased',near_plane=1e-4,far_plane=1e5,eps2d=.0015)
        a=a[0,...,0]
        rgb1+=image[0]+(1-a[...,None])*fills[i]
        rgb2+=image[0]+(1-a[...,None])*A_samples[i]
        alpha+=a
    return dict(F1=rgb1/count,F2=rgb2/count,alpha=alpha/count)
