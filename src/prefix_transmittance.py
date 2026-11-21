"""Per-pixel transmittance before a supplied axial surface.

Uses the same gsplat projection, AA compensation, tile depth sort and alpha
cutoff as the sampled renderer. Historical diagnostic entry stays unchanged.
"""
import math
import torch
from torch.nn import functional as F
import triton as tr
import triton.language as tl


@tr.jit
def _prefix(M,C,Z,O,OFFSET,IDS,LIMIT,OUT,TILES,W:tl.constexpr,H:tl.constexpr,
            GW:tl.constexpr,PROJ_GW:tl.constexpr,SCALE:tl.constexpr,SPARSE:tl.constexpr,
            B:tl.constexpr=32):
    tile=tl.program_id(0)
    if SPARSE:
        tile=tl.load(TILES+tile)
    tx,ty=tile%GW,tile//GW
    tile=(ty//SCALE)*PROJ_GW+tx//SCALE
    p=tl.arange(0,256)
    px=tx*16+p%16;py=ty*16+p//16
    good=(px<W)&(py<H);limit=tl.load(LIMIT+py*W+px,good,other=0.)
    start=tl.load(OFFSET+tile);end=tl.load(OFFSET+tile+1)
    T=tl.full((256,),1.,tl.float32);base=start
    while base<end:
        a=base+tl.arange(0,B);inside=a<end;idx=tl.load(IDS+a,inside,other=0)
        mx=tl.load(M+2*idx,inside,other=0);my=tl.load(M+2*idx+1,inside,other=0)
        c0=tl.load(C+3*idx,inside,other=0);c1=tl.load(C+3*idx+1,inside,other=0);c2=tl.load(C+3*idx+2,inside,other=0)
        z=tl.load(Z+idx,inside,other=0);opacity=tl.load(O+idx,inside,other=0)
        if SCALE == 1:
            dx=mx[None,:]-px[:,None]-.5;dy=my[None,:]-py[:,None]-.5
        else:
            dx=mx[None,:]-(px[:,None]+.5)/SCALE;dy=my[None,:]-(py[:,None]+.5)/SCALE
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
    _prefix[(gw*gh,)](m.contiguous(),conics.contiguous(),z.contiguous(),opacity,offset,flat,
                      depth_limit.contiguous(),out,offset,w,h,gw,gw,1,False,num_warps=8)
    return out


@tr.jit
def _prefix_backward(M,C,Z,O,OFFSET,IDS,LIMIT,T,G,DM,DC,DO,TILES,
                     W:tl.constexpr,H:tl.constexpr,GW:tl.constexpr,PROJ_GW:tl.constexpr,
                     SCALE:tl.constexpr,SPARSE:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0)
    if SPARSE:
        tile=tl.load(TILES+tile)
    tx,ty=tile%GW,tile//GW
    tile=(ty//SCALE)*PROJ_GW+tx//SCALE
    p=tl.arange(0,256)
    px=tx*16+p%16
    py=ty*16+p//16
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
        if SCALE == 1:
            dx=mx[None,:]-px[:,None]-.5
            dy=my[None,:]-py[:,None]-.5
        else:
            dx=mx[None,:]-(px[:,None]+.5)/SCALE
            dy=my[None,:]-(py[:,None]+.5)/SCALE
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
    def forward(ctx, means2d, conics, depth, opacity, offsets, indices, limit, tiles, scale):
        h,w=limit.shape
        gw,gh=math.ceil(w/16),math.ceil(h/16)
        projection_gw=math.ceil((w//scale)/16)
        sparse=tiles is not None
        groups=len(tiles) if sparse else gw*gh
        out=torch.ones_like(limit) if sparse else torch.empty_like(limit)
        tile_ids=tiles if sparse else offsets
        _prefix[(groups,)](means2d,conics,depth,opacity,offsets,indices,limit,out,tile_ids,
                          w,h,gw,projection_gw,scale,sparse,num_warps=8)
        ctx.save_for_backward(means2d,conics,depth,opacity,offsets,indices,limit,out,tile_ids)
        ctx.grid=(groups,w,h,gw,projection_gw,scale,sparse)
        return out

    @staticmethod
    def backward(ctx, upstream):
        means,conics,depth,opacity,offsets,indices,limit,out,tile_ids=ctx.saved_tensors
        groups,*grid=ctx.grid
        dm,dc,do=(torch.zeros_like(v) for v in (means,conics,opacity))
        _prefix_backward[(groups,)](means,conics,depth,opacity,offsets,indices,limit,out,
                                    upstream.contiguous(),dm,dc,do,tile_ids,*grid,num_warps=8)
        # A strict z < limit step has zero derivative away from a crossing.
        # Axial means still receive the projection and AA compensation gradients.
        return dm,dc,None,do,None,None,None,None,None


def differentiable_prefix_transmittance(scene,K,view,depth_limit,eps2d=.0015,far_plane=1e5,
                                         *, active=None, sample_scale=1):
    """Hard prefix on native rays, optionally projected on a coarser surfel grid."""
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode

    if depth_limit.ndim!=2 or depth_limit.dtype!=torch.float32 or scene.means.dtype!=torch.float32:
        raise ValueError('Depth cutoff and Gaussian inputs must be float32 with an H×W cutoff')
    if not len(scene.means):
        return torch.ones_like(depth_limit)+scene.means.sum()*0
    h,w=depth_limit.shape
    if h%sample_scale or w%sample_scale:
        raise ValueError('Ray sampling must tile the projection grid exactly')
    ph,pw=h//sample_scale,w//sample_scale
    gw,gh=math.ceil(pw/16),math.ceil(ph/16)
    tiles=None
    if active is not None:
        # Tile culling only. Every selected pixel retains its own native cutoff.
        used=F.max_pool2d(active[None,None].float(),16,ceil_mode=True).flatten()
        tiles=used.nonzero().flatten().to(torch.int32)
        if not len(tiles):
            return torch.ones_like(depth_limit)+scene.means.sum()*0
        depth_limit=torch.where(active,depth_limit,0.)
    radii,m,z,c,comp=fully_fused_projection(scene.means,None,scene.quats,scene.scales,view[None],K[None],
        pw,ph,eps2d=eps2d,near_plane=1e-4,far_plane=far_plane,calc_compensations=True)
    with torch.no_grad():
        _,ids,flat=isect_tiles(m,radii,z,16,gw,gh)
        offsets=isect_offset_encode(ids,1,gw,gh).reshape(-1)
        offsets=torch.cat((offsets,offsets.new_tensor([len(flat)])))
    opacity=scene.opacities[None]*comp
    return _DifferentiablePrefix.apply(m.contiguous(),c.contiguous(),z.contiguous(),opacity.contiguous(),
                                      offsets,flat,depth_limit.contiguous(),tiles,sample_scale)
