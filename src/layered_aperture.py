"""Shared O5/O6 lens integration: depth-sorted layers, then residual fill.

Layer IDs in GaussianScene.source_frames are provenance codes 0=visible and
1=hidden in this Oracle, not actual input-frame IDs. Scene units are meters.
"""
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from .aperture_renderer import GaussianScene,aperture_disk,render_aperture


def merge_layers(visible: GaussianScene, hidden: GaussianScene = None):
    if hidden is None:return visible
    return GaussianScene(*(torch.cat([getattr(visible,k),getattr(hidden,k)],0) for k in visible.__dataclass_fields__))


def pixel_layer(rgb_linear: Tensor, depth: Tensor, K: Tensor, c2w: Tensor,
                valid: Tensor = None, layer_id: int = 0, sigma_pixels: float = .7,
                opacity: float = .999, subdivisions: int = 5):
    """Pixel-centered, camera-facing surfels; fixed initialization, no fitting.

    Backprojection/splatting are discrete approximations, not exact optics.
    Poses are fixed GT inputs here; the aperture integrator stays differentiable.
    """
    h,w=depth.shape;device,dtype=depth.device,depth.dtype
    if subdivisions>1:
        # Piecewise constant source pixels subdivided in position, not a new
        # image interpolation or fitted color model. Preserve depth boundaries.
        depth=depth.repeat_interleave(subdivisions,0).repeat_interleave(subdivisions,1)
        rgb_linear=rgb_linear.repeat_interleave(subdivisions,0).repeat_interleave(subdivisions,1)
        if valid is not None:valid=valid.repeat_interleave(subdivisions,0).repeat_interleave(subdivisions,1)
    hh,ww=depth.shape
    yy,xx=torch.meshgrid(torch.arange(hh,device=device,dtype=dtype),torch.arange(ww,device=device,dtype=dtype),indexing='ij')
    rays=torch.stack([(xx+.5)/subdivisions,(yy+.5)/subdivisions,torch.ones_like(xx)],-1)@torch.linalg.inv(K).T
    good=torch.isfinite(depth)&(depth>0)&torch.isfinite(rgb_linear).all(-1)
    if valid is not None:good&=valid
    camera=(rays*depth[...,None])[good]
    means=camera@c2w[:3,:3].T+c2w[:3,3]
    sigma=depth[good]/K[0,0]*(sigma_pixels/subdivisions)
    scales=torch.stack([sigma,sigma,sigma*.01],-1)
    quat=Rotation.from_matrix(c2w[:3,:3].detach().cpu().double().numpy()).as_quat()[[3,0,1,2]]
    quats=torch.tensor(np.array(quat),device=device,dtype=dtype)[None].expand(len(means),4).contiguous()
    return GaussianScene(means,quats,scales,torch.full_like(sigma,opacity),rgb_linear[good],
                         torch.full((len(means),),layer_id,device=device,dtype=torch.long))


def aperture_fill_samples(fill_linear: Tensor, background_depth: Tensor, K: Tensor,
                          focus_distance, aperture_radius, samples: int):
    """Intersect each lens ray with local background axial depth, reproject.

    background_depth is a geometry-Oracle input shared by O5/O6. No-hit sky is
    approximated at 10km and separately marked in the returned validity map.
    Border sampling is shared and explicit. LaMa is never an opaque layer.
    """
    h,w=background_depth.shape;device,dtype=background_depth.device,background_depth.dtype
    known=torch.isfinite(background_depth)&(background_depth>0)
    z=torch.where(known,background_depth,torch.full_like(background_depth,1e4))
    focus=torch.as_tensor(focus_distance,device=device,dtype=dtype)
    if focus.ndim or not bool(torch.isfinite(focus)) or float(focus)<=0:raise ValueError('Positive scalar axial focus required')
    yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=dtype),torch.arange(w,device=device,dtype=dtype),indexing='ij')
    disk=aperture_disk(samples,aperture_radius,device=device,dtype=dtype)
    shifts=disk[:,None,None,:]*torch.stack([K[0,0],K[1,1]])*(z.reciprocal()-focus.reciprocal())[None,...,None]
    uv=torch.stack([xx+.5,yy+.5],-1)[None]+shifts
    grid=torch.stack([2*uv[...,0]/w-1,2*uv[...,1]/h-1],-1)
    colors=F.grid_sample(fill_linear.permute(2,0,1)[None].expand(samples,3,h,w),grid,
                         mode='bilinear',padding_mode='border',align_corners=False).permute(0,2,3,1)
    inside=(uv[...,0]>=0)&(uv[...,0]<w)&(uv[...,1]>=0)&(uv[...,1]<h)
    return colors,known,inside


def render_layered(visible, hidden, K, world_to_camera, width, height,
                   focus_distance, aperture_radius, samples, fill_linear,
                   background_depth, return_samples=False, eps2d=.0015):
    """One merged depth sort per aperture camera, followed by T_all*C_fill."""
    fills,known,inside=aperture_fill_samples(fill_linear,background_depth,K,focus_distance,aperture_radius,samples)
    result=render_aperture(merge_layers(visible,hidden),K,world_to_camera,width,height,
        focus_distance,aperture_radius,samples,current_frame=0,
        sample_backgrounds=fills,return_samples=return_samples,eps2d=eps2d)
    result['fill_background_depth_known']=known
    result['fill_projection_inside_fraction']=inside.float().mean(0)
    result['layer_codes']={'visible':0,'hidden':1}
    return result
