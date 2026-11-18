"""Per-pixel transmittance before a supplied axial surface.

Uses the same gsplat projection, AA compensation, tile depth sort and alpha
cutoff as the sampled renderer. Historical diagnostic entry stays unchanged.
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


@tr.jit
def _prefix_backward(M,C,Z,O,OFFSET,IDS,LIMIT,T,G,DM,DC,DO,
                     W:tl.constexpr,H:tl.constexpr,GW:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0)
    p=tl.arange(0,256)
    px=(tile%GW)*16+p%16
    py=(tile//GW)*16+p//16
    good=(px<W)&(py<H)
    pix=py*W+px
    limit=tl.load(LIMIT+pix,good,other=0.)
    upstream=tl.load(G+pix,good,other=0.)*tl.load(T+pix,good,other=0.)
    base=tl.load(OFFSET+tile)
    end=tl.load(OFFSET+tile+1)
    while base<end:
        a=base+tl.arange(0,B)
        inside=a<end
        idx=tl.load(IDS+a,inside,other=0)
        mx=tl.load(M+2*idx,inside,other=0.)
        my=tl.load(M+2*idx+1,inside,other=0.)
        c0=tl.load(C+3*idx,inside,other=0.)
        c1=tl.load(C+3*idx+1,inside,other=0.)
        c2=tl.load(C+3*idx+2,inside,other=0.)
        z=tl.load(Z+idx,inside,other=0.)
        opacity=tl.load(O+idx,inside,other=0.)
        dx=mx[None,:]-px[:,None]-.5
        dy=my[None,:]-py[:,None]-.5
        exponent=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        kernel=tl.exp(-tl.maximum(exponent,0.))
        raw=opacity[None,:]*kernel
        active=inside[None,:]&good[:,None]&(exponent>=0)&(raw>=1/255)&(raw<.999)&(z[None,:]<limit[:,None])
        da=tl.where(active,-upstream[:,None]/(1-tl.minimum(.999,raw)),0.)
        ds=-da*raw
        tl.atomic_add(DO+idx,tl.sum(da*kernel,0),inside)
        tl.atomic_add(DM+2*idx,tl.sum(ds*(c0[None,:]*dx+c1[None,:]*dy),0),inside)
        tl.atomic_add(DM+2*idx+1,tl.sum(ds*(c2[None,:]*dy+c1[None,:]*dx),0),inside)
        tl.atomic_add(DC+3*idx,tl.sum(ds*.5*dx*dx,0),inside)
        tl.atomic_add(DC+3*idx+1,tl.sum(ds*dx*dy,0),inside)
        tl.atomic_add(DC+3*idx+2,tl.sum(ds*.5*dy*dy,0),inside)
        base+=B


class _DifferentiablePrefix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, depth, opacity, offsets, indices, limit):
        h,w=limit.shape
        gw,gh=math.ceil(w/16),math.ceil(h/16)
        out=torch.empty_like(limit)
        _prefix[(gw*gh,)](means2d,conics,depth,opacity,offsets,indices,limit,out,w,h,gw,num_warps=8)
        ctx.save_for_backward(means2d,conics,depth,opacity,offsets,indices,limit,out)
        return out

    @staticmethod
    def backward(ctx, upstream):
        means,conics,depth,opacity,offsets,indices,limit,out=ctx.saved_tensors
        h,w=limit.shape
        gw,gh=math.ceil(w/16),math.ceil(h/16)
        dm,dc,do=(torch.zeros_like(v) for v in (means,conics,opacity))
        _prefix_backward[(gw*gh,)](means,conics,depth,opacity,offsets,indices,limit,out,
                                    upstream.contiguous(),dm,dc,do,w,h,gw,num_warps=8)
        # A strict z < limit step has zero derivative away from a crossing.
        # Axial means still receive the projection and AA compensation gradients.
        return dm,dc,None,do,None,None,None


def differentiable_prefix_transmittance(scene,K,view,depth_limit,eps2d=.0015,far_plane=1e5):
    """Exact hard prefix. Gradients pass through projected Gaussian parameters."""
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode

    if depth_limit.ndim!=2 or depth_limit.dtype!=torch.float32 or scene.means.dtype!=torch.float32:
        raise ValueError('Depth cutoff and Gaussian inputs must be float32 with an H×W cutoff')
    if not len(scene.means):
        return torch.ones_like(depth_limit)+scene.means.sum()*0
    h,w=depth_limit.shape
    gw,gh=math.ceil(w/16),math.ceil(h/16)
    radii,m,z,c,comp=fully_fused_projection(scene.means,None,scene.quats,scene.scales,view[None],K[None],
        w,h,eps2d=eps2d,near_plane=1e-4,far_plane=far_plane,calc_compensations=True)
    with torch.no_grad():
        _,ids,flat=isect_tiles(m,radii,z,16,gw,gh)
        offsets=isect_offset_encode(ids,1,gw,gh).reshape(-1)
        offsets=torch.cat((offsets,offsets.new_tensor([len(flat)])))
    opacity=scene.opacities[None]*comp
    return _DifferentiablePrefix.apply(m.contiguous(),c.contiguous(),z.contiguous(),opacity.contiguous(),
                                      offsets,flat,depth_limit.contiguous())
