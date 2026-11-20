"""Evaluation-only opacity closure at a covered b=0 visible surface."""
import math

import torch
import triton as tr
import triton.language as tl
from gsplat import rasterization

from .ovg_leakage import _product
from .ovg_optics import raster, subcamera


def subpixel_flags(valid, b0, subdivisions):
    repeat = lambda x: x.repeat_interleave(subdivisions, 0).repeat_interleave(subdivisions, 1)
    return repeat(b0)[repeat(valid)].contiguous()


def reproject_coverage(depth, b0, K, focus, offset):
    """Rasterize the unit pixel square centered on each eligible parent.

    Aperture translations preserve axial depth and projected footprint size.
    A tail of a neighboring Gaussian does not close a geometric hole.
    """
    h, w = depth.shape
    y, x = torch.where(b0)
    z = depth[y, x]
    shift = (float(focus)**-1-z.reciprocal())[:, None] * (K.diag()[:2]*offset)
    u = torch.floor(x+.5+shift[:, 0]).long()
    v = torch.floor(y+.5+shift[:, 1]).long()
    ok = (u>=0)&(u<w)&(v>=0)&(v<h)
    cut = depth.new_full((h*w,), torch.inf)
    cut.scatter_reduce_(0, v[ok]*w+u[ok], z[ok], reduce='amin')
    return cut.view(h, w)


@tr.jit
def _close_prefix(M,C,Z,O,RGB,OFF,IDS,IS_B0,CUT,OUT,CLOSED,QW,QZ,Q0,NV,
                  W:tl.constexpr,H:tl.constexpr,GW:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0)
    p=tl.arange(0,256)
    px=(tile%GW)*16+p%16
    py=(tile//GW)*16+p//16
    good=(px<W)&(py<H)
    pix=py*W+px
    cut=tl.load(CUT+pix,good,other=float('inf'))
    cover=good&(cut<float('inf'))
    tol=tl.maximum(cut*2e-6,1e-6)
    start=tl.load(OFF+tile)
    end=tl.load(OFF+tile+1)
    T=tl.full((256,),1.,tl.float32)
    wr=tl.full((256,),0.,tl.float32)
    wg=tl.full((256,),0.,tl.float32)
    wb=tl.full((256,),0.,tl.float32)
    sr=tl.full((256,),0.,tl.float32)
    sg=tl.full((256,),0.,tl.float32)
    sb=tl.full((256,),0.,tl.float32)
    mass=tl.full((256,),0.,tl.float32)
    prefix_mass=tl.full((256,),0.,tl.float32)
    wq=tl.full((256,),0.,tl.float32)
    dq=tl.full((256,),0.,tl.float32)
    if tl.sum(cover.to(tl.int32),0)>0:
        while start<end:
            u=start+tl.arange(0,B)
            active=u<end
            i=tl.load(IDS+u,active,other=0)
            z=tl.load(Z+i,active,other=0)
            mx=tl.load(M+2*i,active,other=0)
            my=tl.load(M+2*i+1,active,other=0)
            a=tl.load(C+3*i,active,other=0)
            b=tl.load(C+3*i+1,active,other=0)
            c=tl.load(C+3*i+2,active,other=0)
            opacity=tl.load(O+i,active,other=0)
            is_b0=tl.load(IS_B0+i,active&(i<NV),other=False)
            dx=mx[None,:]-px[:,None]-.5
            dy=my[None,:]-py[:,None]-.5
            power=.5*(a[None,:]*dx*dx+c[None,:]*dy*dy)+b[None,:]*dx*dy
            alpha=tl.minimum(.999,opacity[None,:]*tl.exp(-power))
            use=active[None,:]&cover[:,None]&(z[None,:]<=cut[:,None]+tol[:,None])
            alpha=tl.where(use&(power>=0)&(alpha>=1/255),alpha,0.)
            rem=1-alpha
            after=tl.associative_scan(rem,1,_product)*T[:,None]
            weights=tl.where(after>1e-4,alpha*after/rem,0.)
            bucket=tl.where(is_b0[None,:]&(tl.abs(z[None,:]-cut[:,None])<=tol[:,None]),weights,0.)
            r=tl.load(RGB+3*i,active,other=0)
            g=tl.load(RGB+3*i+1,active,other=0)
            bcolor=tl.load(RGB+3*i+2,active,other=0)
            wr+=tl.sum(weights*r[None,:],1)
            wg+=tl.sum(weights*g[None,:],1)
            wb+=tl.sum(weights*bcolor[None,:],1)
            sr+=tl.sum(bucket*r[None,:],1)
            sg+=tl.sum(bucket*g[None,:],1)
            sb+=tl.sum(bucket*bcolor[None,:],1)
            mass+=tl.sum(bucket,1)
            prefix_mass+=tl.sum(weights,1)
            q=tl.where(i[None,:]>=Q0,weights,0.)
            wq+=tl.sum(q,1)
            dq+=tl.sum(q*z[None,:],1)
            T=tl.min(after,1)
            start+=B
    close=cover&(mass>0.)
    # Use actual compositing weights: gsplat drops a sample that trips the
    # early-exit threshold, so the raw alpha product is not the RGB remainder.
    remaining_rgb_mass=tl.maximum(1.-prefix_mass,0.)
    remainder=tl.where(close,remaining_rgb_mass/tl.maximum(mass,1e-20),0.)
    tl.store(OUT+3*pix,wr+remainder*sr,good)
    tl.store(OUT+3*pix+1,wg+remainder*sg,good)
    tl.store(OUT+3*pix+2,wb+remainder*sb,good)
    tl.store(CLOSED+pix,close,good)
    tl.store(QW+pix,wq,good)
    tl.store(QZ+pix,tl.where(wq>0,dq/wq,0.),good)


@torch.no_grad()
def opaque_view(scene, K, focus, offset, depth, b0, flags, query_start, background=None):
    """Close the first eligible b=0 surface, preserving earlier compositing.

    Take the original depth-sorted prefix through that surface, then give
    its b=0 fragments the remaining mass in proportion to existing weights.
    No more distant fragment or F2 receives weight on a closed ray.
    """
    h,w=depth.shape
    intr,view=subcamera(K,focus,offset)
    colors,alpha,m=rasterization(scene.means,scene.quats,scene.scales,scene.opacities,scene.colors,
        view[None],intr[None],w,h,packed=False,render_mode='RGB',rasterize_mode='antialiased',
        eps2d=.0015,near_plane=1e-4,far_plane=1e5)
    old=colors[0]
    if background is not None:
        old=old+(1-alpha[0])*background
    cut=reproject_coverage(depth,b0,K,focus,offset)
    out=torch.empty_like(old)
    closed=torch.empty_like(cut,dtype=torch.bool)
    qw=torch.empty_like(cut)
    qz=torch.empty_like(cut)
    gw,gh=math.ceil(w/16),math.ceil(h/16)
    offsets=m['isect_offsets'].flatten()
    ids=m['flatten_ids']
    offsets=torch.cat([offsets,offsets.new_tensor([len(ids)])])
    _close_prefix[(gw*gh,)](m['means2d'].contiguous(),m['conics'].contiguous(),
        m['depths'].contiguous(),m['opacities'].contiguous(),scene.colors.contiguous(),
        offsets,ids,flags,cut,out,closed,qw,qz,query_start,len(flags),w,h,gw,num_warps=8)
    image=torch.where(closed[...,None],out,old)
    return image,dict(closed=closed,depth=cut,query_weight=qw,query_depth=qz)


def render(scene,K,focus,offsets,depth,b0,flags,query_start,background=None,*,opaque_b0=False):
    if not opaque_b0:
        return raster(scene,K,focus,offsets,*depth.shape,background)
    return torch.stack([opaque_view(scene,K,focus,o,depth,b0,flags,query_start,
                        background[i] if background is not None else None)[0]
                        for i,o in enumerate(offsets)])
