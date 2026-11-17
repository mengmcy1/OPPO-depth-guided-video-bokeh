"""Fixed appendix12 3-sigma footprint replacement, without parameter fitting.

Footprints are rasterized at pixel centers with the renderer's AA covariance.
The aperture expansion is an Euclidean distance transform against the method
band. A padded canvas includes footprints centered outside the output image.
"""
import math
import numpy as np
import torch
import triton as tr
import triton.language as tl
from scipy.ndimage import distance_transform_edt


@tr.jit
def _footprints(M,C,Z,OFFSET,IDS,DISTANCE,COC,FLAGS,PIXELS,QUERY,
                W:tl.constexpr,H:tl.constexpr,GW:tl.constexpr,MODE:tl.constexpr,
                B:tl.constexpr=32):
    tile=tl.program_id(0)
    p=tl.arange(0,256)
    x=tile%GW*16+p%16
    y=tile//GW*16+p//16
    valid=(x<W)&(y<H)
    start=tl.load(OFFSET+tile)
    end=tl.load(OFFSET+tile+1)
    base=start
    if MODE==0:
        distance=tl.load(DISTANCE+y*W+x,valid,other=1e10)
    elif MODE==2:
        query=tl.load(QUERY+y*W+x,valid,other=0)
    while base<end:
        j=base+tl.arange(0,B)
        inside=j<end
        idx=tl.load(IDS+j,inside,other=0)
        dx=tl.load(M+2*idx,inside,other=0)[None,:]-x[:,None]-.5
        dy=tl.load(M+2*idx+1,inside,other=0)[None,:]-y[:,None]-.5
        c0=tl.load(C+3*idx,inside,other=0)
        c1=tl.load(C+3*idx+1,inside,other=0)
        c2=tl.load(C+3*idx+2,inside,other=0)
        q=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        hit=valid[:,None]&inside[None,:]&(q>=0)&(q<=4.5)
        if MODE==0:
            radius=tl.load(COC+idx,inside,other=0)
            chosen=hit&(distance[:,None]<=radius[None,:])
            tl.atomic_max(FLAGS+idx,1,inside&(tl.sum(chosen.to(tl.int32),0)>0),sem='relaxed')
        elif MODE==1:
            chosen=hit&tl.load(FLAGS+idx,inside,other=0)[None,:]
            tl.atomic_max(PIXELS+y*W+x,1,valid&(tl.sum(chosen.to(tl.int32),1)>0),sem='relaxed')
        else:
            chosen=hit&query[:,None]
            tl.atomic_max(FLAGS+idx,1,inside&(tl.sum(chosen.to(tl.int32),0)>0),sem='relaxed')
        base+=B


@torch.no_grad()
def replacement_footprints(scene,K,band,p2_focus,aperture=.05/2.8):
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode
    height,width=band.shape
    z=scene.means[:,2]
    coc=K[0,0]*aperture*(z.clamp_min(1e-4).reciprocal()-1/p2_focus).abs()
    coc=torch.where(z>1e-4,coc,torch.zeros_like(coc)).contiguous()
    padding=int(torch.ceil(coc.max()).item())+4
    if padding>2048:
        raise RuntimeError('Footprint padding exceeds2048px; report before expanding resource scope')
    W,H=width+2*padding,height+2*padding
    intr=K.clone()
    intr[:2,2]+=padding
    r,m,d,c,_=fully_fused_projection(scene.means,None,scene.quats,scene.scales,
        torch.eye(4,device=K.device)[None],intr[None],W,H,eps2d=.0015,
        near_plane=1e-4,far_plane=1e5,calc_compensations=True)
    gw,gh=math.ceil(W/16),math.ceil(H/16)
    _,ii,flat=isect_tiles(m,r,d,16,gw,gh)
    offset=isect_offset_encode(ii,1,gw,gh).flatten()
    offset=torch.cat([offset,torch.tensor([len(flat)],device=K.device,dtype=offset.dtype)])
    padded=np.pad(band, padding)
    distance=distance_transform_edt(~padded).astype(np.float32)
    cv=lambda a:torch.as_tensor(a,device=K.device)
    flags=torch.zeros(len(scene.means),device=K.device,dtype=torch.int32)
    pixels=torch.zeros((H,W),device=K.device,dtype=torch.int32)
    overlaps=torch.zeros_like(flags)
    args=(m.contiguous(),c.contiguous(),d.contiguous(),offset,flat,cv(distance),coc)
    if band.any():
        _footprints[(gw*gh,)](*args,flags,pixels,pixels,W,H,gw,0,num_warps=8)
        _footprints[(gw*gh,)](*args,flags,pixels,pixels,W,H,gw,1,num_warps=8)
    footprint=pixels[padding:padding+height,padding:padding+width].bool().cpu().numpy()
    coverage=band|footprint
    query=cv(np.pad(coverage,padding))
    _footprints[(gw*gh,)](*args,overlaps,pixels,query,W,H,gw,2,num_warps=8)
    deleted=flags.bool()
    overlap=overlaps.bool()&~deleted
    return deleted,coverage,overlap,dict(padding_px=padding,max_Gaussian_CoC_px=float(coc.max()),
        footprint='pixel-center 3sigma ellipse using native AA covariance; Euclidean band distance <= continuous CoC radius; padded frame',
        deleted=int(deleted.sum()),retained_overlapping=int(overlap.sum()),coverage_pixels=int(coverage.sum()),coverage_fraction=float(coverage.mean()))


def subset_scene(scene,keep):
    """Actually remove selected output Gaussians; preserve retained attributes."""
    from .aperture_renderer import GaussianScene
    return GaussianScene(**{k:getattr(scene,k)[keep] for k in scene.__dataclass_fields__})


@tr.jit
def _hidden_depth(GI,START,COUNT,PIX,M,C,Z,O,REF,OUT,
                  W:tl.constexpr,N:tl.constexpr,HIDDEN_BEGIN,L:tl.constexpr):
    pixel=tl.load(PIX+tl.program_id(0))
    start=tl.load(START+pixel)
    n=tl.load(COUNT+pixel)
    j=tl.arange(0,L)
    idx=tl.load(GI+start+j,j<n,other=0)
    dx=tl.load(M+2*idx)-pixel%W-.5
    dy=tl.load(M+2*idx+1)-pixel//W-.5
    sigma=.5*(tl.load(C+3*idx)*dx*dx+tl.load(C+3*idx+2)*dy*dy)+tl.load(C+3*idx+1)*dx*dy
    a=tl.where(j<n,tl.minimum(.999,tl.load(O+idx)*tl.exp(-sigma)),0.)
    w=a*tl.cumprod(1-a,0)/tl.maximum(1-a,1e-8)
    w=tl.where((j<n)&(idx>=HIDDEN_BEGIN),w,0.)
    z=tl.load(Z+idx)
    ref=tl.load(REF+pixel)
    relative=(z-ref)/tl.maximum(ref,1e-8)
    maximum=tl.max(w,0)
    # Stable tie by first native depth-sorted hit; no GT in this selection.
    at=tl.min(tl.where((w==maximum)&(w>0),j,2147483647),0)
    dom=tl.sum(tl.where(j==at,relative,0.),0)
    total=tl.sum(w,0)
    tl.store(OUT+pixel,total)
    tl.store(OUT+N+pixel,dom)
    tl.store(OUT+2*N+pixel,tl.sum(w*relative,0)/tl.maximum(total,1e-30))
    tl.store(OUT+3*N+pixel,tl.sum(tl.where(relative<-.03,w,0.),0))
    tl.store(OUT+4*N+pixel,tl.sum(tl.where(tl.abs(relative)<=.03,w,0.),0))
    tl.store(OUT+5*N+pixel,tl.sum(tl.where(relative>.03,w,0.),0))
    tl.store(OUT+6*N+pixel,tl.sum(tl.where(relative<0,w,0.),0))


@torch.no_grad()
def hidden_depth_contributions(scene,K,hidden_begin,reference_depth,selection):
    """Exact center-view composited hidden weights; no RGB image rendering.

    Returns total hidden weight, strongest-hidden-Gaussian relative depth,
    weight-mean relative depth, three depth-bin masses and strictly-front mass.
    """
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range
    from .local_gaussian_replacement import projection
    h,w=selection.shape
    m,z,c,o,offset,flat=projection(scene,K,width=w,height=h)
    gi,pi,_=rasterize_to_indices_in_range(0,100000,torch.ones((1,h,w),device=K.device),m,c,o,w,h,16,offset,flat)
    count=torch.bincount(pi.long(),minlength=h*w).contiguous()
    start=(count.cumsum(0)-count).contiguous()
    out=torch.zeros((7,h*w),device=K.device)
    selected=torch.as_tensor(selection,device=K.device).flatten()
    ref=torch.as_tensor(reference_depth,device=K.device).flatten().contiguous()
    last=0
    largest=int(count.max())
    for exponent in range(4,max(4,math.ceil(math.log2(max(largest,1))))+1):
        length=2**exponent
        pixels=torch.nonzero(selected&(count>last)&(count<=length)).flatten().contiguous()
        if len(pixels):
            _hidden_depth[(len(pixels),)](gi,start,count,pixels,m,c,z,o,ref,out,w,h*w,hidden_begin,length,num_warps=4 if length<=128 else 8)
        last=length
    return out.reshape(7,h,w).cpu().numpy()
