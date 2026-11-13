"""Diagnostic-only per-pixel transmittance before a supplied axial surface.

Uses the same gsplat projection, AA compensation, tile depth sort and alpha
cutoff as the sampled renderer. No colors, fitting or method behavior change.
"""
import math
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _prefix(M,C,Z,O,OFFSET,IDS,LIMIT,OUT,W:tl.constexpr,H:tl.constexpr,
            GW:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0);p=tl.arange(0,256)
    px=(tile%GW)*16+p%16;py=(tile//GW)*16+p//16
    good=(px<W)&(py<H);limit=tl.load(LIMIT+py*W+px,good,other=0.)
    start=tl.load(OFFSET+tile);end=tl.load(OFFSET+tile+1)
    T=tl.full((256,),1.,tl.float32);base=start
    while base<end:
        a=base+tl.arange(0,B);inside=a<end;idx=tl.load(IDS+a,inside,other=0)
        mx=tl.load(M+2*idx,inside,other=0);my=tl.load(M+2*idx+1,inside,other=0)
        c0=tl.load(C+3*idx,inside,other=0);c1=tl.load(C+3*idx+1,inside,other=0);c2=tl.load(C+3*idx+2,inside,other=0)
        z=tl.load(Z+idx,inside,other=0);opacity=tl.load(O+idx,inside,other=0)
        dx=mx[None,:]-px[:,None]-.5;dy=my[None,:]-py[:,None]-.5
        sigma=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        alpha=tl.minimum(.999,opacity[None,:]*tl.exp(-sigma))
        use=inside[None,:]&good[:,None]&(sigma>=0)&(alpha>=1/255)&(z[None,:]<limit[:,None])
        T*=tl.exp(tl.sum(tl.where(use,tl.log(1-alpha),0.),1));base+=B
    tl.store(OUT+py*W+px,T,good)


@torch.no_grad()
def prefix_transmittance(scene,K,view,depth_limit,eps2d=.0015,far_plane=1e5):
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode
    h,w=depth_limit.shape;gw,gh=math.ceil(w/16),math.ceil(h/16)
    radii,m,z,conics,comp=fully_fused_projection(scene.means,None,scene.quats,scene.scales,view[None],K[None],w,h,eps2d=eps2d,near_plane=1e-4,far_plane=far_plane,calc_compensations=True)
    opacity=(scene.opacities[None]*comp).contiguous()
    _,ids,flat=isect_tiles(m,radii,z,16,gw,gh);offset=isect_offset_encode(ids,1,gw,gh).reshape(-1)
    offset=torch.cat([offset,torch.tensor([len(flat)],device=offset.device,dtype=offset.dtype)])
    out=torch.empty_like(depth_limit)
    _prefix[(gw*gh,)](m.contiguous(),conics.contiguous(),z.contiguous(),opacity,offset,flat,depth_limit.contiguous(),out,w,h,gw,num_warps=8)
    return out
